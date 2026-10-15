"""Guards for the P2 / P3 / P4 changes.

Written as unittest.TestCase (not bare pytest functions) because CI runs
`python -m unittest discover tests`, which does not collect pytest-style
functions. A guard that CI never executes is not a guard.

Each test encodes a failure mode that was actually observed, so a future
refactor that reintroduces it fails CI instead of silently regressing accuracy
or speed.
"""
import unittest

import torch
import torch.nn.functional as F

from mgn.blocks import ConvAML_18Layer_Hybrid
from mgn.blocks.hybrid_router import HybridBlock
from train_continual_aml import FisherEWC


def _model(**kw):
    kw.setdefault("K", 6)
    kw.setdefault("max_layers", 18)
    kw.setdefault("max_dim", 32)
    kw.setdefault("num_classes", 10)
    return ConvAML_18Layer_Hybrid(**kw)


def _manual_trunk(model, x, gates, block_mode, dim_masks):
    """Replicates the pre-fix loop: run every block, then blend by the gate."""
    h = model.stem(x)
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            w = block_mode[:, i]
            d = dim_masks[:, i].view(-1, model.max_dim, 1, 1)
            g = gates[:, i:i + 1].view(-1, 1, 1, 1)
            hc = layer(h, mode_weights=w, dim_mask=d)
            h = hc if i == 0 else g * hc + (1.0 - g) * h
            if i in (5, 11):
                h = F.max_pool2d(h, 2, 2)
    return h


def _logits(model, feat, pi):
    hp = model.ln(model.pool(feat).flatten(1))
    wo = (pi @ model.proto_weights[:, :model.wo_size]).view(-1, model.num_classes, model.head_dim)
    bo = pi @ model.proto_weights[:, model.wo_size:model.wo_size + model.bo_size]
    return (hp.unsqueeze(1) @ wo.transpose(1, 2)).squeeze(1) + bo


def _drift_a_shared_param(model):
    """Perturb a weight EWC actually constrains.

    Perturbing model.parameters()[0] is wrong: the first registered parameter is
    structure_prototypes, which _shared_trunk deliberately excludes, so the
    penalty would stay at zero for reasons unrelated to EWC.
    """
    shared = FisherEWC._shared_trunk(model)
    n, p = next(iter(shared.items()))
    with torch.no_grad():
        p.add_(1.0)
    return n


class TestDepthShortCircuit(unittest.TestCase):
    """P2 - dynamic depth must short-circuit, not multiply a computed result by 0."""

    def test_closed_gate_is_exact_identity(self):
        torch.manual_seed(0)
        m = _model().eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            _, st, pi, _ = m(x, tau=0.1)
        g = st.layer_gates.clone()
        for i in (2, 3, 7, 12):
            g[:, i] = 0.0
        with torch.no_grad():
            out, _, pi_r, diag = m(x, tau=0.1, struct_override=(g, st.block_mode))
            feat = _manual_trunk(m, x, g, st.block_mode, st.dim_masks)
        self.assertTrue(torch.allclose(_logits(m, feat, pi_r), out, atol=1e-5))
        # layer 0 is exempt from gating, so 4 of 18 are skipped
        self.assertEqual(diag["n_executed"], 18 - 4)

    def test_per_sample_skip_does_not_leak_uninitialised_memory(self):
        """Regression: scattering into torch.empty_like handed closed-gate samples
        whatever happened to be in that memory."""
        torch.manual_seed(0)
        m = _model().eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            _, st, _, _ = m(x, tau=0.1)
        g = st.layer_gates.clone()
        g[2:, 5] = 0.0
        with torch.no_grad():
            out, _, pi, _ = m(x, tau=0.1, struct_override=(g, st.block_mode))
            feat = _manual_trunk(m, x, g, st.block_mode, st.dim_masks)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(_logits(m, feat, pi), out, atol=1e-5))

    def test_soft_gate_is_never_short_circuited(self):
        """Regression: short-circuiting at the 0.5 hard-gate threshold would turn a
        soft gate of 0.2 - which must blend as 0.2*h_cand + 0.8*h - into a plain
        identity, silently changing the routed model's output. Only gate <= eps
        is an exact identity."""
        m = _model().eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            _, st, _, _ = m(x, tau=0.1)
        soft = torch.full((4, 18), 0.2)
        soft[:, 4] = 0.0            # only this one is a true identity
        soft[2:, 9] = 0.0           # per-sample exact identity
        with torch.no_grad():
            out, _, pi, diag = m(x, tau=0.1, struct_override=(soft, st.block_mode))
            feat = _manual_trunk(m, x, soft, st.block_mode, st.dim_masks)
        self.assertTrue(torch.allclose(_logits(m, feat, pi), out, atol=1e-5))
        # layer 4 skipped for all, layer 9 for half, layer 0 never
        self.assertEqual(diag["n_executed"], 17)

    def test_fully_open_gates_execute_everything(self):
        m = _model().eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            _, st, _, diag = m(x, tau=0.1)
        self.assertEqual(diag["n_executed"], 18.0)

    def test_training_never_short_circuits(self):
        """`i > 0 and skip_layer[i]` does not short-circuit when skip_layer is
        None, which crashed every training step before it was made a list."""
        m = _model().train()
        x = torch.randn(2, 3, 32, 32)
        _, _, _, diag = m(x, tau=0.8)
        self.assertEqual(diag["n_executed"], 18.0)
        out, _, _, _ = m(x, tau=0.8)
        out.sum().backward()
        blk = m.layers[5]
        grads = [blk.plain.conv.weight.grad,
                 blk.resnet.conv1.weight.grad,
                 blk.vit.qkv.weight.grad]
        # routing is batch-consistent during training, so only the chosen branch
        # is exercised
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grads))

    def test_architecture_gradient_survives(self):
        m = _model().train()
        out, _, _, _ = m(torch.randn(2, 3, 32, 32), tau=0.8)
        out.sum().backward()
        g = m.block_mode_prototypes.grad
        self.assertIsNotNone(g)
        self.assertGreater(g.abs().sum().item(), 0.0)


class TestPerSampleDispatch(unittest.TestCase):
    """P3 - MoE dispatch is an inference-only optimisation."""

    def test_dispatch_is_numerically_equivalent_for_one_hot(self):
        torch.manual_seed(0)
        blk = HybridBlock(32, 32).eval()
        x = torch.randn(6, 32, 16, 16)
        dm = torch.ones(6, 32, 1, 1)
        mw = torch.eye(3)[[0, 1, 2, 0, 1, 2]]
        with torch.no_grad():
            ref = blk(x, mw, dm)
            idx = mw.argmax(-1)
            got = torch.empty_like(x)
            for i, br in enumerate((blk.plain, blk.resnet, blk.vit)):
                mk = idx == i
                if mk.any():
                    got[mk] = br(x[mk], dim_mask=dm[mk])
        self.assertTrue(torch.allclose(got, ref, atol=1e-5))

    def test_dispatch_keeps_the_architecture_gradient_in_training(self):
        """Regression: out[mask]=branch(x[mask]) drops mode_weights from the
        autograd graph and zeroes the architecture gradient exactly."""
        blk = HybridBlock(32, 32).train()
        x = torch.randn(4, 32, 8, 8)
        dm = torch.ones(4, 32, 1, 1)
        logits = torch.randn(1, 1, 3, requires_grad=True)
        hard = F.gumbel_softmax(logits, tau=0.8, hard=True, dim=-1)
        mw = ((hard - logits.detach()) + logits).expand(4, -1, -1).reshape(4, 3)
        blk(x, mw, dm).pow(2).mean().backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_soft_routing_keeps_the_mixture(self):
        """With non-one-hot weights the dispatch identity does not hold, so the
        soft path must stay reachable."""
        blk = HybridBlock(32, 32).eval()
        x = torch.randn(4, 32, 8, 8)
        dm = torch.ones(4, 32, 1, 1)
        mw = torch.tensor([[0.3, 0.6, 0.1]]).expand(4, -1).contiguous()
        with torch.no_grad():
            out = blk(x, mw, dm)
        self.assertEqual(tuple(out.shape), (4, 32, 8, 8))

    def test_finite_across_batch_sizes_under_autocast(self):
        """Regression: the dispatch buffer was built with torch.empty_like and
        raised a dtype mismatch under bf16 autocast (conv -> bf16, LayerNorm ->
        fp32)."""
        if not torch.cuda.is_available():
            self.skipTest("no CUDA")
        m = _model().to("cuda").eval()
        with torch.no_grad():
            for bs in (1, 3, 8):
                x = torch.randn(bs, 3, 32, 32, device="cuda")
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out, _, _, _ = m(x, tau=0.1)
                self.assertTrue(torch.isfinite(out).all(), f"bs={bs}")


class TestArchitecturePrior(unittest.TestCase):

    def test_staged_prior_is_the_measured_default(self):
        """Neutral was measured to lose ~79 accuracy points at the tested budget;
        see experiments/dynamic_depth_routing_2026-10-14.md section 6."""
        self.assertEqual(_model().arch_prior, "staged")

    def test_staged_prior_layout(self):
        m = _model(arch_prior="staged")
        bm = m.block_mode_prototypes.detach()
        self.assertGreater(bm[:, 0:6, 1].mean().item(), bm[:, 0:6, 2].mean().item())
        self.assertGreater(bm[:, 12:18, 2].mean().item(), bm[:, 12:18, 0].mean().item())

    def test_neutral_prior_uses_noise_not_exact_zeros(self):
        """Exact zeros make argmax tie-break to index 0 and pin all 18 layers to
        PlainConv at task 0."""
        m = _model(arch_prior="neutral").eval()
        self.assertGreater(m.block_mode_prototypes.detach().std().item(), 0.0)
        with torch.no_grad():
            _, st, _, _ = m(torch.randn(4, 3, 32, 32), tau=0.1)
        modes = st.block_mode.argmax(-1)
        self.assertFalse(torch.equal(modes, torch.zeros_like(modes)))


class TestFlopsModel(unittest.TestCase):

    def test_vit_flops_match_the_runtime_token_cap(self):
        """compute_flops() must use the pooled token count forward() actually
        runs, otherwise a complexity loss penalises the wrong mode by ~76x."""
        blk = HybridBlock(32, 32)
        f32 = blk.compute_flops((32, 32), mode_idx=2)
        f8 = blk.compute_flops((8, 8), mode_idx=2)
        self.assertLess(f32, 4.0 * f8,
                        f"vit @32x32 is {f32/1e6:.1f}M, @8x8 is {f8/1e6:.1f}M")

    def test_vit_is_cheapest_only_where_token_pooling_applies(self):
        """At 32x32 and 16x16 ViT pools to 64 tokens and is by far the cheapest.
        At 8x8 there is nothing to pool, so Plain wins and the claim "ViT is the
        cheapest mode everywhere" is false."""
        blk = HybridBlock(48, 48)
        self.assertLess(blk.compute_flops((32, 32), mode_idx=2),
                        blk.compute_flops((32, 32), mode_idx=0))
        self.assertLess(blk.compute_flops((16, 16), mode_idx=2),
                        blk.compute_flops((16, 16), mode_idx=1))
        self.assertGreater(blk.compute_flops((8, 8), mode_idx=2),
                           blk.compute_flops((8, 8), mode_idx=0))

    def test_cost_term_is_opt_in(self):
        """The differentiable cost term costs 54 extra CUDA launches per forward
        (+16 % latency measured), so it must not run unless requested."""
        m = _model().train()
        _, _, _, diag = m(torch.randn(2, 3, 32, 32), tau=0.8)
        self.assertNotIn("expected_gflops", diag)
        self.assertNotIn("expected_mflops", diag)

        m.compute_cost = True
        out, _, _, diag = m(torch.randn(2, 3, 32, 32), tau=0.8)
        self.assertIn("expected_gflops", diag)
        (out.sum() + diag["expected_gflops"]).backward()
        g = m.block_mode_prototypes.grad
        self.assertIsNotNone(g)
        self.assertGreater(g.abs().sum().item(), 0.0)

    def test_flops_term_is_normalised_for_the_loss(self):
        """Regression: 0.01 x 638 MFLOP added ~6.4 nats to a ~2.3 cross-entropy and
        stopped the model classifying at all."""
        m = _model(max_dim=48).train()
        m.compute_cost = True
        _, _, _, diag = m(torch.randn(2, 3, 32, 32), tau=0.8)
        self.assertLess(abs(diag["expected_gflops"].item()), 5.0)
        self.assertGreater(diag["expected_mflops"].item(), 10.0)
        # the loss actually sees GFLOPs, i.e. mflops / 1000
        self.assertAlmostEqual(diag["expected_gflops"].item(),
                               diag["expected_mflops"].item() / 1e3, places=5)


class TestFisherEWC(unittest.TestCase):

    def _one_task(self, e, m):
        out, _, _, _ = m(torch.randn(2, 3, 32, 32), tau=0.8)
        out.pow(2).mean().backward()
        e.observe_batch()
        e.consolidate()

    def test_penalty_is_zero_until_the_weights_drift(self):
        m = _model(max_dim=16)
        e = FisherEWC(m, ewc_lambda=10.0)
        self._one_task(e, m)
        self.assertTrue(e._fisher_all)
        self.assertTrue(all(v.numel() > 0 for v in e._fisher_all.values()))
        # one task, so B/A == theta* exactly in exact arithmetic; the residual
        # is float32 round-off, not a behavioural difference
        self.assertAlmostEqual(e.penalty().item(), 0.0, places=9)
        _drift_a_shared_param(m)
        self.assertGreater(e.penalty().item(), 0.0)

    def test_penalty_survives_a_second_task(self):
        """Regression: consolidate() replaced the Fisher instead of folding it in,
        so the penalty was identically zero for every task after the first."""
        m = _model(max_dim=16)
        e = FisherEWC(m, ewc_lambda=10.0)
        for _ in range(2):
            self._one_task(e, m)
        self.assertTrue(e._fisher_all, "Fisher must accumulate ACROSS tasks")
        _drift_a_shared_param(m)
        self.assertGreater(e.penalty().item(), 0.0)

    def test_excluded_parameters(self):
        shared = set(FisherEWC._shared_trunk(_model(max_dim=16)).keys())
        self.assertFalse([n for n in shared if "proto_weights" in n])
        self.assertFalse([n for n in shared if "block_mode_prototypes" in n])
        self.assertFalse([n for n in shared if "structure_prototypes" in n])
        self.assertTrue([n for n in shared if "layers" in n])

    def test_disabled_is_a_noop(self):
        m = _model(max_dim=16)
        e = FisherEWC(m, ewc_lambda=0.0)
        self._one_task(e, m)
        self.assertFalse(e._fisher_all)
        self.assertEqual(e.penalty().item(), 0.0)

    def test_parameters_without_gradient_are_skipped_not_zero_filled(self):
        """Routing is batch-consistent, so only the chosen branch of each
        HybridBlock gets a gradient. Recording those as zero-Fisher would anchor
        them to theta* forever."""
        m = _model(max_dim=16)
        e = FisherEWC(m, ewc_lambda=1.0)
        out, _, _, _ = m(torch.randn(2, 3, 32, 32), tau=0.8)
        out.pow(2).mean().backward()
        e.observe_batch()
        self.assertTrue(all(v.abs().sum() > 0 for v in e._fisher.values()))
        self.assertLess(set(e._fisher.keys()), set(FisherEWC._shared_trunk(m).keys()))

    def test_state_is_one_vector_not_one_per_task(self):
        m = _model(max_dim=16)
        e = FisherEWC(m, ewc_lambda=1.0)
        before = e.state_bytes()
        for _ in range(3):
            self._one_task(e, m)
        # Fisher + theta* only, no growth with the number of tasks
        self.assertLessEqual(e.state_bytes(), before + 2 * 1_500_000)


class TestOpenSetFallback(unittest.TestCase):
    """--auto-live-thresh is an open-set fallback for the AUTONOMOUS protocol."""

    def _tiny_loader(self, n=8):
        x = torch.randn(n, 3, 32, 32)
        y = torch.randint(0, 10, (n,))
        return [(x, y)]

    def test_disabled_by_default_and_reported(self):
        from train_continual_aml import evaluate
        m = _model(max_dim=16).eval()
        dev = torch.device("cpu")
        acc, fp = evaluate(m, self._tiny_loader(), dev)
        self.assertIn("live_fallback_frac", fp)
        self.assertEqual(fp["live_fallback_frac"], 0.0)
        self.assertGreaterEqual(acc, 0.0)

    def test_signature_defaults_to_off(self):
        import inspect

        from train_continual_aml import evaluate
        sig = inspect.signature(evaluate)
        self.assertEqual(sig.parameters["auto_live_thresh"].default, 0.0)
        self.assertEqual(sig.parameters["amb_topk"].default, 0)

    def test_wrong_width_struct_override_raises_a_clear_error(self):
        """Regression: a gate vector of the wrong width used to surface as an
        opaque IndexError from inside the layer loop."""
        m = _model().eval()
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            _, st, _, _ = m(x, tau=0.1)
        with self.assertRaises(ValueError) as cm:
            m(x, tau=0.1, struct_override=(torch.ones(4, 7), st.block_mode))
        self.assertIn("max_layers", str(cm.exception))

    def test_fallback_does_not_leak_into_the_oracle_protocol(self):
        """The fallback must only apply to the autonomous protocol. With a single
        domain the live weights and the snapshot are identical, so enabling the
        fallback must leave the oracle accuracy bit-identical."""
        from train_continual_aml import evaluate
        m = _model(max_dim=16).eval()
        dev = torch.device("cpu")
        loader = self._tiny_loader()
        snaps = [{
            "enc": {k: v.detach().clone() for k, v in m.vision_encoder.state_dict().items()},
            "sel": {k: v.detach().clone() for k, v in m.prototype_selector.state_dict().items()},
            "stem": {k: v.detach().clone() for k, v in m.stem.state_dict().items()},
            "layers": {k: v.detach().clone() for k, v in m.layers.state_dict().items()},
            "ln": {k: v.detach().clone() for k, v in m.ln.state_dict().items()},
            "struct_heads": {"sprotos": m.structure_prototypes.detach().clone(),
                             "bmprotos": m.block_mode_prototypes.detach().clone()},
            "protos": m.proto_weights.detach().to("cpu", torch.float32).clone(),
        }]
        loader = self._tiny_loader()
        _, st, _, _ = m(loader[0][0], tau=0.1)
        slots = [{"gate": torch.ones(m.max_layers), "bm_mode": st.block_mode[0].clone(),
                  "bm_res": st.block_mode[0, :, 1].clone()}]

        class _Head(torch.nn.Module):
            def __init__(self, n=1):
                super().__init__()
                self.l = torch.nn.Linear(m.latent_dim, n)

            def forward(self, z):
                return self.l(z)

        head = _Head()
        acc_off, _ = evaluate(m, loader, dev, dom_head=head, slots=slots,
                              snaps=snaps, route_tau=0.6, dom_truth=0,
                              auto_live_thresh=0.0)
        acc_on, _ = evaluate(m, loader, dev, dom_head=head, slots=slots,
                             snaps=snaps, route_tau=0.6, dom_truth=0,
                             auto_live_thresh=0.5)
        self.assertEqual(acc_off, acc_on,
                         "fallback must not alter the task-oracle protocol")


class TestAmbiguousRetryBudget(unittest.TestCase):
    """The ambiguous-sample retry is the real inference cost: it re-loads
    enc + sel + stem + 18 layers + ln per domain per batch."""

    def test_zero_keeps_try_every_domain(self):
        from train_continual_aml import ambiguous_candidate_mask
        pd = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        amb = torch.tensor([True, True])
        m = ambiguous_candidate_mask(pd, amb, 3, 0)
        self.assertEqual(m.tolist(), [True, True, True])

    def test_topk_restricts_the_retry(self):
        from train_continual_aml import ambiguous_candidate_mask
        pd = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]])
        amb = torch.tensor([True, True, True])
        m = ambiguous_candidate_mask(pd, amb, 3, 1)
        # every sample's argmax is retained
        self.assertEqual(m.tolist(), [True, True, True])
        m2 = ambiguous_candidate_mask(pd, amb, 3, 2)
        # union of top-2 across the three samples = all three domains here
        self.assertEqual(m2.tolist(), [True, True, True])

    def test_topk_actually_drops_a_domain(self):
        from train_continual_aml import ambiguous_candidate_mask
        pd = torch.tensor([[0.8, 0.15, 0.05], [0.85, 0.10, 0.05]])
        amb = torch.tensor([True, True])
        self.assertEqual(ambiguous_candidate_mask(pd, amb, 3, 1).tolist(),
                         [True, False, False])
        self.assertEqual(ambiguous_candidate_mask(pd, amb, 3, 2).tolist(),
                         [True, True, False])

    def test_no_ambiguous_samples_keeps_every_domain(self):
        from train_continual_aml import ambiguous_candidate_mask
        pd = torch.tensor([[0.8, 0.15, 0.05], [0.85, 0.10, 0.05]])
        amb = torch.tensor([False, False])
        self.assertEqual(ambiguous_candidate_mask(pd, amb, 3, 1).tolist(),
                         [True, True, True])

    def test_topk_larger_than_domain_count_is_clamped(self):
        from train_continual_aml import ambiguous_candidate_mask
        pd = torch.tensor([[0.8, 0.15, 0.05]])
        amb = torch.tensor([True])
        self.assertEqual(ambiguous_candidate_mask(pd, amb, 3, 99).tolist(),
                         [True, True, True])

    def test_topk_cannot_scatter_past_the_snapshot_count(self):
        """Regression: pd indexes STRUCTURE SLOTS and the scatter destination
        indexes SNAPSHOTS. With more slots than snapshots a top-k column
        >= n_domains was an out-of-bounds scatter_ (RuntimeError on CPU, and a
        device-side assert that kills the CUDA context on GPU)."""
        from train_continual_aml import ambiguous_candidate_mask
        pd = torch.tensor([[0.05, 0.05, 0.4, 0.5]])          # 4 slots
        amb = torch.tensor([True])
        # only 2 snapshots exist -> the two best slots (3,2) have no snapshot
        m = ambiguous_candidate_mask(pd, amb, 2, 2)
        self.assertEqual(m.shape, torch.Size([2]))
        self.assertEqual(m.tolist(), [True, True])           # best VALID slots
        # a single snapshot with a decisive slot-0 preference
        pd2 = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
        self.assertEqual(ambiguous_candidate_mask(pd2, amb, 2, 1).tolist(),
                         [True, False])
        # n_domains == 0 must not raise either
        self.assertEqual(ambiguous_candidate_mask(pd, amb, 0, 2).tolist(), [])

    def test_data_cache_key_includes_the_dataset_root(self):
        """Regression: _DATA_CACHE was keyed on the dataset name alone, so a
        second get_task_data() for the same name with a different data_root in
        the same process silently returned the first root's tensors."""
        import inspect
        import train_continual_aml as tca
        src = inspect.getsource(tca.get_task_data)
        self.assertIn("os.path.abspath(data_root)", src)
        self.assertNotIn("_DATA_CACHE[name]", src)


class TestNormalizationChoices(unittest.TestCase):
    """--norm previously advertised bn/gn1/in but only ever built two of them."""

    def test_in_builds_instance_norm_not_group_norm(self):
        """Regression: the binary `if bn: BatchNorm else: GroupNorm` branch plus
        the trainer's `args.norm != 'bn' -> 'gn'` mapping meant `--norm in`
        silently produced GroupNorm."""
        import torch.nn as nn
        from mgn.blocks.cnn_block import PlainConvBlock, ResNetBlock, make_norm
        self.assertIsInstance(make_norm("in", 8), nn.InstanceNorm2d)
        self.assertIsInstance(make_norm("gn1", 8), nn.GroupNorm)
        self.assertIsInstance(make_norm("bn", 8), nn.BatchNorm2d)
        self.assertIsInstance(PlainConvBlock(8, 8, norm_type="in").norm,
                              nn.InstanceNorm2d)
        self.assertIsInstance(ResNetBlock(8, 8, norm_type="in").norm1,
                              nn.InstanceNorm2d)

    def test_every_norm_type_is_distinct_and_supported(self):
        from mgn.blocks.cnn_block import make_norm
        for t in ("bn", "gn1", "in"):
            self.assertIsNotNone(make_norm(t, 8))
        with self.assertRaises(ValueError):
            make_norm("bogus", 8)

    def test_norm_types_forward_at_every_stage_resolution(self):
        import torch
        from mgn.blocks.cnn_block import ResNetBlock
        for t in ("bn", "gn1", "in"):
            for hw in (32, 16, 8):
                blk = ResNetBlock(16, 16, norm_type=t)
                out = blk(torch.randn(2, 16, hw, hw))
                self.assertEqual(tuple(out.shape), (2, 16, hw, hw), f"{t}@{hw}")

    def test_cli_help_is_runnable(self):
        """Regression: argparse help strings containing a literal `%` (e.g.
        '89.9%') are treated as format strings, so `--help` crashed with
        ValueError: unsupported format character."""
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "train_continual_aml.py", "--help"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[-400:])
        self.assertIn("--norm", r.stdout)

    def test_default_norm_is_instance(self):
        """bn carries running statistics that go stale per candidate branch;
        measured A[1,1] 84.75 vs 98.03 for InstanceNorm."""
        import subprocess
        import sys
        r = subprocess.run([sys.executable, "train_continual_aml.py", "--help"],
                           capture_output=True, text=True)
        i = r.stdout.rindex("--norm")   # the detailed entry, not the usage line
        # the help text is wrapped, so match the marker rather than the
        # "(default: ...)" suffix which argparse may push onto a later line
        self.assertIn("DEFAULT CHANGED bn -> in", r.stdout[i:i + 900])

    def test_every_block_default_matches_the_cli_default(self):
        """The CLI default is only meaningful if the classes agree.

        `--norm` defaults to `in` (InstanceNorm), and BatchNorm is the only
        option carrying running statistics, which go stale per candidate
        branch. A block class that still defaulted to `bn` builds a *different
        model* for anyone constructing it directly - which is exactly what
        happened here: `cnn_block` and `hybrid_18` were aligned to `in` and
        `hybrid_router.HybridBlock` was missed, so the legacy executor kept
        building BatchNorm blocks while the trainer asked for InstanceNorm.
        """
        import inspect
        from mgn.blocks import cnn_block, hybrid_router
        from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid

        for cls in (cnn_block.PlainConvBlock, cnn_block.ResNetBlock,
                    hybrid_router.HybridBlock):
            sig = inspect.signature(cls.__init__)
            self.assertEqual(sig.parameters["norm_type"].default, "in",
                             f"{cls.__name__} still defaults to BatchNorm")
        sig = inspect.signature(ConvAML_18Layer_Hybrid.__init__)
        self.assertEqual(sig.parameters["norm_type"].default, "in")

    def test_class_default_actually_builds_instance_norm(self):
        """Not just the signature: build the block the way an unaware caller
        would and inspect the module that actually got constructed."""
        import torch.nn as nn
        from mgn.blocks.cnn_block import PlainConvBlock, ResNetBlock
        from mgn.blocks.hybrid_router import HybridBlock
        self.assertIsInstance(PlainConvBlock(4, 4).norm, nn.InstanceNorm2d)
        self.assertIsInstance(ResNetBlock(4, 4).norm1, nn.InstanceNorm2d)
        hb = HybridBlock(4, 4)
        self.assertIsInstance(hb.plain.norm, nn.InstanceNorm2d)
        self.assertIsInstance(hb.resnet.norm1, nn.InstanceNorm2d)

    def test_max_dim_below_the_width_floor_is_not_silent(self):
        """Depth gating has `gate_skip_eps`; width gating has a floor of 16.

        That floor is a clamp on `d_lengths = clamp(sigmoid(...) * max_dim,
        min=16.0)`, so when `max_dim < 16` the clamp binds for every sample and
        `dim_masks` saturates at ~1.0: the width gate does nothing at all. The
        library tests build `max_dim=8` models, and neither they nor a user
        passing a smaller `--max-dim` got any signal that one of the three
        routing dimensions had become inert.
        """
        import warnings
        import torch
        from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            m = ConvAML_18Layer_Hybrid(K=4, max_layers=2, max_dim=8,
                                       num_classes=10).eval()
        self.assertTrue(any("width gate is inert" in str(w.message) for w in caught),
                        "a max_dim under the width floor must say so")

        with torch.no_grad():
            _, struct, _, _ = m(torch.randn(2, 3, 32, 32), tau=0.1)
        dm = struct.dim_masks
        self.assertTrue(bool((dm > 0.999).all()),
                        "the width mask is saturated, so width routing is inert")
        # and dim_weights, which reports effective width, exceeds 1
        self.assertGreater(float(struct.dim_weights.max()), 1.0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConvAML_18Layer_Hybrid(K=4, max_layers=2, max_dim=48, num_classes=10)
        self.assertFalse(any("width gate is inert" in str(w.message) for w in caught),
                         "the shipped max_dim must stay quiet")

    def test_train_hybrid_norm_matches_the_cli_default(self):
        """Two scripts, one architecture. `train_hybrid_aml.py` constructed
        `HybridBlock(in, in, num_heads)` without `norm_type` and hard-coded
        `nn.GroupNorm` in its stem, so it trained a BatchNorm/GroupNorm network
        while `train_continual_aml.py --norm in` trained an InstanceNorm one --
        with nothing in either output saying so. Assert the built modules, not
        the arguments, since the bug was an argument that was never passed.
        """
        import torch.nn as nn
        from train_hybrid_aml import HybridAMLNetwork

        m = HybridAMLNetwork(num_classes=10, max_layers=2, hidden_dim=8,
                             K=3, num_heads=2)
        self.assertIsInstance(m.stem[1], nn.InstanceNorm2d)
        self.assertIsInstance(m.layers[0].plain.norm, nn.InstanceNorm2d)
        self.assertIsInstance(m.layers[0].resnet.norm1, nn.InstanceNorm2d)
        # and the comparison remains reachable, since that is the experiment
        b = HybridAMLNetwork(num_classes=10, max_layers=2, hidden_dim=8,
                             K=3, num_heads=2, norm_type="bn")
        self.assertIsInstance(b.stem[1], nn.BatchNorm2d)
        self.assertIsInstance(b.layers[0].plain.norm, nn.BatchNorm2d)


if __name__ == "__main__":
    unittest.main()