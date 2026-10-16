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


if __name__ == "__main__":
    unittest.main()
