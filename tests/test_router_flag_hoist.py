"""Equivalence guards for the hoisted routing flags.

`ConvAML_18Layer_Hybrid.forward` now reads every layer's
`(all_hard, same_mode, chosen)` in one device->host transfer
(`_hoist_block_mode_flags`) instead of letting each `HybridBlock` synchronise
twice. The entire justification for the change is that it is a latency change and
nothing else, so that is what these tests pin:

  * the batched predicate equals the per-layer predicate, including the mixed
    (per-sample) case and the near-one-hot case that the `> 0.99` threshold was
    written for;
  * `HybridBlock.forward(x, mode_weights, flags=...)` is bitwise equal to
    `HybridBlock.forward(x, mode_weights)`;
  * the full 18-layer model is bitwise equal with the hoist and with the old
    per-layer path restored by monkeypatch, in both training and eval mode.

unittest.TestCase, not bare pytest functions: CI runs
`python -m unittest discover tests` and does not collect pytest-style functions.
"""
import unittest

import torch

from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid, _hoist_block_mode_flags
from mgn.blocks.hybrid_router import HybridBlock


def _per_layer_flags(mode_weights):
    """The reference: exactly what `HybridBlock.forward` computes on its own."""
    mode_max, mode_idx = mode_weights.max(dim=-1)
    flags = torch.stack(((mode_max > 0.99).all(), (mode_idx == mode_idx[0]).all()))
    all_hard, same_mode = flags.tolist()
    return bool(all_hard), bool(same_mode), int(mode_idx[0].item())


def _small_model(seed=3, max_layers=4, max_dim=8, classes=4, K=4):
    torch.manual_seed(seed)
    return ConvAML_18Layer_Hybrid(
        K=K, latent_dim=32, max_layers=max_layers, max_dim=max_dim,
        num_classes=classes, num_heads=2, norm_type="gn1", encoder_tail_act="none")


class TestHoistMatchesPerLayer(unittest.TestCase):
    """The batched predicate must agree with the per-layer one everywhere."""

    def _assert_same(self, block_mode):
        hoisted = _hoist_block_mode_flags(block_mode)
        for i in range(block_mode.shape[1]):
            self.assertEqual(
                hoisted[i], _per_layer_flags(block_mode[:, i]),
                msg=f"layer {i} disagrees: {hoisted[i]} vs "
                    f"{_per_layer_flags(block_mode[:, i])}")

    def test_batch_consistent_training_style(self):
        """Training samples one mode per layer for the whole batch."""
        B, L = 8, 5
        bm = torch.zeros(B, L, 3)
        for i in range(L):
            bm[:, i, i % 3] = 1.0
        self._assert_same(bm)
        flags = _hoist_block_mode_flags(bm)
        self.assertTrue(all(f[0] and f[1] for f in flags))
        self.assertEqual([f[2] for f in flags], [0, 1, 2, 0, 1])

    def test_mixed_per_sample_eval_style(self):
        """Eval takes an argmax per sample, so `same_mode` can differ by layer."""
        torch.manual_seed(0)
        bm = torch.eye(3)[torch.randint(0, 3, (12, 7))].float()
        self._assert_same(bm)
        flags = _hoist_block_mode_flags(bm)
        self.assertTrue(all(f[0] for f in flags), "one-hot rows are always hard")
        self.assertTrue(any(not f[1] for f in flags),
                        "a random batch should disagree on some layer")

    def test_threshold_boundary(self):
        """0.99 is the boundary the original predicate used; keep it exactly."""
        bm = torch.tensor([[[0.995, 0.0, 0.005]], [[0.995, 0.0, 0.005]]])
        self.assertEqual(_hoist_block_mode_flags(bm), [(True, True, 0)])
        bm = torch.tensor([[[0.985, 0.01, 0.005]], [[0.985, 0.01, 0.005]]])
        self.assertEqual(_hoist_block_mode_flags(bm), [(False, True, 0)])

    def test_struct_override_soft_rows(self):
        """struct_override supplies soft per-domain slots, not one-hot rows."""
        torch.manual_seed(1)
        bm = torch.softmax(torch.randn(4, 6, 3), dim=-1)
        self._assert_same(bm)
        self.assertTrue(all(not f[0] for f in _hoist_block_mode_flags(bm)))


class TestBlockForwardEquivalence(unittest.TestCase):
    """Passing `flags` must not change one bit of the block's output."""

    def _block(self):
        torch.manual_seed(0)
        return HybridBlock(8, 8, num_heads=2, norm_type="gn1")

    def _compare(self, blk, x, mw, dim_mask):
        with torch.no_grad():
            ref = blk(x, mode_weights=mw, dim_mask=dim_mask)
            got = blk(x, mode_weights=mw, dim_mask=dim_mask,
                      flags=_per_layer_flags(mw))
        self.assertTrue(torch.equal(ref, got),
                        "flags= changed the block output")

    def test_eval_same_mode_all_branches(self):
        blk = self._block().eval()
        x = torch.randn(4, 8, 8, 8)
        dm = torch.rand(4, 8, 1, 1)
        for mode in range(3):
            mw = torch.zeros(4, 3)
            mw[:, mode] = 1.0
            self._compare(blk, x, mw, dm)

    def test_eval_mixed_modes(self):
        blk = self._block().eval()
        x = torch.randn(6, 8, 8, 8)
        mw = torch.eye(3)[torch.tensor([0, 1, 2, 2, 0, 1])].float()
        self._compare(blk, x, mw, None)
        self._compare(blk, x, mw, torch.rand(6, 8, 1, 1))

    def test_training_mode(self):
        blk = self._block().train()
        x = torch.randn(4, 8, 8, 8)
        mw = torch.zeros(4, 3)
        mw[:, 1] = 1.0
        self._compare(blk, x, mw, torch.rand(4, 8, 1, 1))

    def test_soft_mixture_path(self):
        """Soft rows fall through to the DARTS mixture; flags must agree there too."""
        blk = self._block().train()
        x = torch.randn(4, 8, 8, 8)
        mw = torch.softmax(torch.randn(4, 3), dim=-1)
        self._compare(blk, x, mw, None)


class TestModelForwardEquivalence(unittest.TestCase):
    """End to end: the hoisted model equals the model with the old path forced."""

    @staticmethod
    def _without_hoist():
        """Context manager-ish pair: patch HybridBlock.forward to ignore `flags`."""
        orig = HybridBlock.forward

        def ignore_flags(self, x, mode_weights=None, dim_mask=None, flags=None):
            return orig(self, x, mode_weights=mode_weights, dim_mask=dim_mask,
                        flags=None)

        return orig, ignore_flags

    def _run_both(self, model, x, pi_mask, seed):
        orig, ignore_flags = self._without_hoist()
        HybridBlock.forward = ignore_flags
        try:
            torch.manual_seed(seed)
            a = model(x, tau=0.8, pi_mask=pi_mask)
        finally:
            HybridBlock.forward = orig
        torch.manual_seed(seed)
        b = model(x, tau=0.8, pi_mask=pi_mask)
        return a, b

    def test_eval_mode_bitwise_equal(self):
        model = _small_model().eval()
        x = torch.randn(3, 3, 16, 16)
        pi_mask = torch.zeros(4, dtype=torch.bool)
        pi_mask[:2] = True
        with torch.no_grad():
            (out_a, _, pi_a, diag_a), (out_b, _, pi_b, diag_b) = self._run_both(
                model, x, pi_mask, seed=11)
        self.assertTrue(torch.equal(out_a, out_b), "eval output differs")
        self.assertTrue(torch.equal(pi_a, pi_b), "eval routing differs")
        self.assertEqual(diag_a["n_executed"], diag_b["n_executed"])

    def test_training_mode_bitwise_equal(self):
        """Training samples its architecture from the global RNG, so the hoist
        must not consume RNG differently -- identical seed must give identical
        architecture and identical output."""
        model = _small_model().train()
        x = torch.randn(3, 3, 16, 16)
        pi_mask = torch.zeros(4, dtype=torch.bool)
        pi_mask[:2] = True
        (out_a, _, pi_a, diag_a), (out_b, _, pi_b, diag_b) = self._run_both(
            model, x, pi_mask, seed=5)
        self.assertTrue(torch.equal(out_a, out_b), "train output differs")
        self.assertTrue(torch.equal(pi_a, pi_b), "train routing differs")
        self.assertTrue(torch.equal(diag_a["layer_soft"], diag_b["layer_soft"]))
        self.assertTrue(torch.equal(diag_a["bm_logits"], diag_b["bm_logits"]))

    def test_gradients_bitwise_equal(self):
        """The backward pass must be unaffected as well."""
        model = _small_model().train()
        x = torch.randn(3, 3, 16, 16)
        orig, ignore_flags = self._without_hoist()

        def grads():
            model.zero_grad(set_to_none=True)
            torch.manual_seed(5)
            out, _, _, _ = model(x, tau=0.8)
            out.float().pow(2).mean().backward()
            return {k: p.grad.detach().clone()
                    for k, p in model.named_parameters() if p.grad is not None}

        HybridBlock.forward = ignore_flags
        try:
            g_old = grads()
        finally:
            HybridBlock.forward = orig
        g_new = grads()
        self.assertEqual(set(g_old), set(g_new))
        for k in g_old:
            self.assertTrue(torch.equal(g_old[k], g_new[k]),
                            f"gradient of {k} differs")


class TestSoftTailPredicateHoist(unittest.TestCase):
    """The soft-mixture branch predicates: one transfer instead of three.

    The soft tail of `HybridBlock.forward` -- the path taken when `all_hard` is
    False, i.e. evaluation under per-domain SOFT slots -- gated each branch on
    `(w > 1e-4).any()`. Each of those is a CUDA tensor read by a Python `if`, so
    three of them per layer is 54 device->host syncs per 18-layer forward, each
    draining the launch queue. They now come from one `.any(dim=0).tolist()`.

    As with the flag hoist above, the whole claim is that this is a latency
    change and nothing else, so these tests pin the three things that could make
    it more than that: the decisions must be identical (including the corner
    where a branch is inactive, which is the only case where the two forms could
    differ in *how many* branches run), the outputs must be bitwise equal, and
    the number of transfers must actually drop.
    """

    SHIPPED = HybridBlock.forward

    def _soft_tail_reference(self, x, mode_weights, dim_mask):
        """The pre-hoist body, verbatim, used as the equivalence oracle."""
        w_plain = mode_weights[:, 0].view(-1, 1, 1, 1)
        w_resnet = mode_weights[:, 1].view(-1, 1, 1, 1)
        w_vit = mode_weights[:, 2].view(-1, 1, 1, 1)
        out_plain = self.plain(x, dim_mask=dim_mask) if (w_plain > 1e-4).any() else 0.0
        out_resnet = self.resnet(x, dim_mask=dim_mask) if (w_resnet > 1e-4).any() else 0.0
        out_vit = self.vit(x, dim_mask=dim_mask) if (w_vit > 1e-4).any() else 0.0
        return w_plain * out_plain + w_resnet * out_resnet + w_vit * out_vit

    def _with_reference_tail(self, fn):
        """Run `fn` with the pre-hoist soft tail installed."""
        def reference_forward(self, blk_x, mode_weights=None, dim_mask=None,
                              flags=None):
            if mode_weights is None or mode_weights.ndim != 2:
                return TestSoftTailPredicateHoist.SHIPPED(
                    self, blk_x, mode_weights, dim_mask, flags)
            if flags is not None and flags[0]:
                return TestSoftTailPredicateHoist.SHIPPED(
                    self, blk_x, mode_weights, dim_mask, flags)
            return TestSoftTailPredicateHoist._soft_tail_reference(
                self, blk_x, mode_weights, dim_mask)
        return reference_forward

    def _model(self, seed=5, layers=4):
        torch.manual_seed(seed)
        return ConvAML_18Layer_Hybrid(
            K=4, latent_dim=32, max_layers=layers, max_dim=8, num_classes=4,
            num_heads=2, norm_type="gn1", encoder_tail_act="none").eval()

    def _soft_override(self, n, layers, device, gates=0.9, mode=(0.5, 0.4, 0.1)):
        g = torch.full((n, layers), gates, device=device)
        b = torch.tensor(mode, device=device).expand(n, layers, 3)
        return (g.contiguous(), b.contiguous())

    # Cover the activation patterns that can actually occur. `block_mode` is a
    # softmax slot, so every row sums to 1 and at least one entry is >= 1/3: a
    # row with NO branch above the threshold is not reachable from the trainer or
    # from `evaluate`.
    #
    # It is deliberately absent from this list. In that state both forms skip
    # every branch and return `w_plain * 0.0 + ...`, i.e. a [B, 1, 1, 1] tensor
    # rather than [B, C, H, W] -- the mixture has no branch to supply a shape.
    # That is pre-existing behaviour of the shipped tail and the hoist does not
    # change it, but it means the comparison cannot be run there, so it is
    # excluded instead of quietly asserted.
    CASES = {
        "all_active": (0.5, 0.4, 0.1),
        "plain_only": (0.9, 1e-6, 1e-6),
        "resnet_only": (1e-6, 0.9, 1e-6),
        "vit_only": (1e-6, 1e-6, 0.9),
        "plain_and_vit": (0.45, 1e-6, 0.45),
        "at_threshold": (1e-4, 1e-4, 0.9),
        "just_below": (9.9e-5, 9.9e-5, 0.9),
    }

    def test_block_mode_rows_are_normalised(self):
        """The precondition the case list above relies on, asserted rather than assumed.

        `evaluate` overrides `block_mode` with per-domain softmax slots, so a row
        that left every branch below 1e-4 would break the tail's shape. Pin the
        invariant that rules it out.
        """
        model = self._model()
        x = torch.randn(6, 3, 32, 32)
        ov = self._soft_override(6, model.max_layers, x.device)
        with torch.no_grad():
            _, struct, _, _ = model(x, tau=0.1, struct_override=ov)
        bm = struct.block_mode
        self.assertTrue(torch.allclose(
            bm.sum(-1), torch.ones_like(bm.sum(-1)), atol=1e-5),
            "block_mode rows must sum to 1")
        self.assertTrue(bool((bm.max(dim=-1).values > 1e-4).all()),
                        "every row must have at least one live branch")

    def _capture(self, tail_forward, mode, n=6, layers=4):
        model = self._model(layers=layers)
        x = torch.randn(n, 3, 32, 32)
        ov = self._soft_override(n, layers, x.device, mode=mode)
        HybridBlock.forward = tail_forward
        try:
            with torch.no_grad():
                out, struct, _, diag = model(x, tau=0.1, struct_override=ov)
        finally:
            HybridBlock.forward = TestSoftTailPredicateHoist.SHIPPED
        return (out.detach().clone(), struct.block_mode.detach().clone(),
                struct.layer_gates.detach().clone(),
                diag["layer_soft"].detach().clone())

    def test_bitwise_equal_to_the_pre_hoist_tail(self):
        """Every activation pattern must produce identical tensors."""
        ref = self._with_reference_tail(None)
        for name, mode in TestSoftTailPredicateHoist.CASES.items():
            new = self._capture(TestSoftTailPredicateHoist.SHIPPED, mode)
            old = self._capture(ref, mode)
            for a, b, tag in zip(new, old, ("output", "block_mode",
                                            "layer_gates", "layer_soft")):
                self.assertTrue(torch.equal(a, b),
                                f"{name}: {tag} differs from the pre-hoist tail")

    def test_the_soft_tail_is_actually_reached(self):
        """Guards the test above from passing vacuously.

        With a soft override `all_hard` is False, so the model must fall through
        to the mixture rather than take the single-branch fast path.
        """
        model = self._model()
        x = torch.randn(6, 3, 32, 32)
        ov = self._soft_override(6, model.max_layers, x.device)
        with torch.no_grad():
            _, struct, _, _ = model(x, tau=0.1, struct_override=ov)
        mx = struct.block_mode.max(dim=-1).values
        self.assertFalse(bool((mx > 0.99).all()),
                         "override must not be one-hot, or the soft tail is skipped")

    def test_transfer_count_drops_from_three_to_one_in_the_tail(self):
        """The saving is the whole point, so assert it rather than describe it.

        The reference here replaces only the soft-mixture body and reaches it
        directly, so it never runs the `(all_hard, same_mode)` transfer that
        precedes it: its 3 are the 3 branch predicates and nothing else. The
        shipped forward does pay that flags transfer when the caller does not
        supply `flags`, hence 2 rather than 1 there. Production does supply them
        -- the caller hoists all 18 layers' triples in one transfer -- which is
        the case pinned at the bottom.
        """
        block = HybridBlock(in_channels=8, out_channels=8, num_heads=2,
                            norm_type="gn1").eval()
        x = torch.randn(4, 8, 4, 4)
        dim_mask = torch.ones(4, 8, 1, 1)
        mw = torch.tensor([[0.5, 0.4, 0.1]] * 4)

        def count_transfers(fn, **kw):
            """Count device->host syncs.

            Two different spellings of the same thing have to be counted: the
            pre-hoist predicates are `(w > 1e-4).any()` consumed by a Python
            `if`, which syncs through `Tensor.__bool__`, while the hoisted form
            is a single `.tolist()`. Counting only one of them would make the
            comparison meaningless.
            """
            calls = {"n": 0}
            real_tolist = torch.Tensor.tolist
            real_bool = torch.Tensor.__bool__

            def counting_tolist(t):
                calls["n"] += 1
                return real_tolist(t)

            def counting_bool(t):
                calls["n"] += 1
                return real_bool(t)

            torch.Tensor.tolist = counting_tolist
            torch.Tensor.__bool__ = counting_bool
            HybridBlock.forward = fn
            try:
                with torch.no_grad():
                    fn(block, x, mode_weights=mw, dim_mask=dim_mask, **kw)
            finally:
                torch.Tensor.tolist = real_tolist
                torch.Tensor.__bool__ = real_bool
                HybridBlock.forward = TestSoftTailPredicateHoist.SHIPPED
            return calls["n"]

        shipped = TestSoftTailPredicateHoist.SHIPPED
        ref = self._with_reference_tail(None)

        self.assertEqual(count_transfers(ref), 3,
                         "pre-hoist tail: one transfer per branch predicate")
        self.assertEqual(count_transfers(shipped), 2,
                         "shipped, flags not supplied: flags transfer + 1 tail")
        self.assertEqual(count_transfers(shipped, flags=(False, True, 0)), 1,
                         "production shape: flags hoisted by the caller, so one "
                         "transfer per layer")
        self.assertEqual(count_transfers(ref, flags=(False, True, 0)), 3,
                         "pre-hoist tail still pays 3 even with flags supplied")


if __name__ == "__main__":
    unittest.main()
