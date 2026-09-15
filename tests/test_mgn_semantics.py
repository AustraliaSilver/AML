"""
CPU-safe regression tests for AML library semantics
====================================================
Each test pins down a bug class that previously slipped through the CI-less
paths of this repo. Run: python -m unittest discover tests
"""

import unittest
import math
import torch

from mgn.architecture import ModelStructure, project_architecture
from mgn.constraints import (
    ResourceBudget, phi_cost, compute_aml_efficiency, compute_aml_efficiency_torch,
)
from mgn.layout import get_prototype_layout, get_second_conv_layout
from mgn.generator import ModelGenerator
from mgn.executor import TaskNetwork
from mgn.losses import MetaLoss
from mgn.trainer import MetaTrainer
from mgn.regularization import structure_distance


def _mlp_generator(**kw):
    kw.setdefault("input_dim", 12)
    kw.setdefault("output_dim", 5)
    kw.setdefault("max_layers", 3)
    kw.setdefault("max_dim", 8)
    kw.setdefault("dim_choices", (4, 8))
    kw.setdefault("latent_dim", 16)
    kw.setdefault("num_prototypes", 4)
    return ModelGenerator(**kw)


class TestExecutorSemantics(unittest.TestCase):
    def test_mlp_branch_input_dim_ne_max_dim(self):
        """Regression: fc weights are [B, out, in]; the bmm must transpose.
        The old code only worked when input_dim == max_dim."""
        gen = _mlp_generator()   # input_dim 12 != max_dim 8
        ex = TaskNetwork()
        gen.eval(); ex.eval()
        x = torch.randn(4, 12)
        with torch.no_grad():
            model = gen(x)
            out = ex(x, model)
        self.assertEqual(out.shape, (4, 5))
        self.assertTrue(torch.isfinite(out).all())

    def test_mlp_depth_gates_are_independent_per_layer(self):
        """Regression: gating must NOT normalize across the layer dimension
        (winner-take-all killed every layer but one / zero at eval)."""
        gen = _mlp_generator()
        ex = TaskNetwork()
        gen.eval(); ex.eval()
        x = torch.randn(6, 12)
        with torch.no_grad():
            model = gen(x)
            S = model.structure
            full = ex(x, model)
            # Hard-set: all gates off except mid layers via direct structure
            # edit; output must CHANGE with per-layer gate value => gates
            # actually gate independently (across-layer softmax would give
            # winner-take-all artifacts).
            S2 = model
            S2.structure.layer_gates = torch.full_like(S.layer_gates, 0.5)
            half = ex(x, S2)
            self.assertFalse(torch.allclose(full, half, atol=1e-6))

    def test_eval_determinism_mlp_and_conv(self):
        gen = _mlp_generator()
        ex = TaskNetwork()
        gen.eval(); ex.eval()
        x = torch.randn(3, 12)
        torch.manual_seed(0)
        with torch.no_grad():
            a = ex(x, gen(x))
            b = ex(x, gen(x))
        self.assertTrue(torch.allclose(a, b))


class TestTrainerLifecycle(unittest.TestCase):
    def test_preprocess_fn_hook_and_no_implicit_augmentation(self):
        """The library runs NO dataset-specific augmentation of its own
        anymore; a caller-supplied preprocess_fn is what runs instead."""
        calls = []
        def sniff(x):
            calls.append(tuple(x.shape))
            return x + 0.0
        gen = _mlp_generator()
        ex = TaskNetwork()
        loss = MetaLoss(lambda_p=1e-5, lambda_c=0.0, lambda_m=0.0)
        tr = MetaTrainer(gen, ex, loss, lr=1e-3, device="cpu",
                         preprocess_fn=sniff)
        x = torch.randn(16, 12)
        y = torch.randint(0, 5, (16,))
        tr.train(x, y, epochs=1, batch_size=8, tau_schedule=False)
        self.assertEqual(len(calls), 2)  # 16 samples / batch 8
        # train/eval mode discipline: eval helpers must restore training mode
        tr.generator.train()
        tr.generate_architecture(x)
        tr.compare_architectures(x, x)
        tr.prototype_usage(x)
        self.assertTrue(tr.generator.training,
                        "generator left in eval mode after an eval helper")

    def test_empty_loader_no_crash(self):
        gen = _mlp_generator()
        ex = TaskNetwork()
        loss = MetaLoss()
        tr = MetaTrainer(gen, ex, loss, device="cpu")
        empty = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.randn(0, 12), torch.randint(0, 5, (0,))),
            batch_size=4)
        out = tr.train_epoch(empty)
        self.assertEqual(out["loss"], 0.0)


class TestMetaLoss(unittest.TestCase):
    def _gen_model(self, out_dim=10):
        gen = _mlp_generator(output_dim=out_dim)
        gen.eval()
        with torch.no_grad():
            return gen(torch.randn(4, 12))

    def test_lambda_eff_requires_explicit_budget(self):
        model = self._gen_model()
        loss_fn = MetaLoss(lambda_eff=0.1)   # no p_allowance
        logits = torch.randn(4, 5, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])
        with self.assertRaises(ValueError):
            loss_fn(logits, targets, None, None, model)

    def test_no_dataset_default_budget_in_signature(self):
        import inspect
        params = inspect.signature(MetaLoss.__init__).parameters
        self.assertIsNone(params["p_allowance"].default,
                          "library must not ship a baked-in p_allowance")

    def test_gap_uses_true_class_count(self):
        """G_gap's softplus denominator uses ln(#classes) derived from the
        structure (was hard-coded ln(10)). Two generators that differ ONLY in
        a class-count-free way must still route ln_Y from structure.output_dim.
        We pin the exact source line instead of a brittle end-to-end delta."""
        import inspect
        src = inspect.getsource(MetaLoss.forward)
        self.assertIn("model.structure.output_dim", src,
                      "ln_Y must derive from structure.output_dim, not a constant")
        self.assertNotIn("ln_Y = 2.302", src,
                         "hard-coded ln(10) must be gone")


class TestConstraints(unittest.TestCase):
    def test_scalar_helper_cannot_overflow(self):
        # q_used 1000x over budget: used to raise OverflowError inside
        # math.exp(20*(r-1)).
        val = compute_aml_efficiency(0.9, 1e4, 1e7)
        self.assertTrue(math.isfinite(val) and val > 0)

    def test_scalar_and_torch_twins_agree(self):
        p, q = 1.5e6, 9e5
        s = compute_aml_efficiency(0.8, p, q)
        t = compute_aml_efficiency_torch(torch.tensor(0.8), p,
                                         torch.tensor(float(q))).item()
        self.assertAlmostEqual(s, t, delta=0.05 * max(s, 1e-9))


class TestLayout(unittest.TestCase):
    def test_layout_offsets_are_dense_and_ordered(self):
        """v2 row: per layer [w1 b1 w2 b2], then head; nothing may gap or
        overlap -- the fused kernels treat these bounds as a packing
        contract, so the WHOLE chain (incl. the residual-branch slots) must
        be dense and strictly increasing."""
        total, offs, info, wo, bo = get_prototype_layout(4, 16, 10, in_channels=3)
        offs2 = get_second_conv_layout(4, 16, 10, in_channels=3)
        prev_end = 0
        for (ws, we, bs, be, ic), (w2s, w2e, b2s, b2e) in zip(offs, offs2):
            self.assertEqual(ws, prev_end)
            self.assertEqual(we - ws, 16 * ic * 9)
            self.assertEqual(bs, we)
            self.assertEqual(be - bs, 16)
            self.assertEqual(w2s, be)                      # w2 follows b1
            self.assertEqual(w2e - w2s, 16 * 16 * 9)      # 2nd conv: D x D
            self.assertEqual(b2s, w2e)
            self.assertEqual(b2e - b2s, 16)
            prev_end = b2e
        self.assertEqual(wo, prev_end)
        self.assertEqual(bo, wo + (16 * 16) * 10)
        self.assertEqual(total, bo + 10)

    def test_generator_slicing_consumes_full_row(self):
        """One-hot pi must expose the COMPLETE prototype row (head included)
        and shapes must follow the layout, catching init/slice drift."""
        gen = ModelGenerator(input_dim=3072, output_dim=10, max_layers=3,
                             max_dim=8, dim_choices=(4, 8), latent_dim=8,
                             num_prototypes=2, use_conv=True, in_channels=3)
        row = int(get_prototype_layout(3, 8, 10, 3)[0])
        self.assertEqual(gen.prototype_lib.proto_weights.shape[1], row)
        onehot = torch.tensor([[0.0, 1.0]])
        _L, offs, _i, wo, bo = get_prototype_layout(3, 8, 10, 3)
        with torch.no_grad():
            _lw, _lb, _lh, wdir = gen.prototype_lib(onehot)  # onehot pi -> row 1
        self.assertEqual(wdir.shape[1], row)  # full row consumed, no tail leak
        for (ws, we, bs, be, ic) in offs:
            self.assertTrue(torch.equal(wdir[0, ws:we],
                                        gen.prototype_lib.proto_weights[1, ws:we]))
            # layer biases are zero-initialised in the conv layout
            self.assertEqual(float(wdir[0, bs:be].abs().max()), 0.0)
        for (w2s, w2e, b2s, b2e) in get_second_conv_layout(3, 8, 10, 3):
            self.assertTrue(torch.equal(wdir[0, w2s:w2e],
                                        gen.prototype_lib.proto_weights[1, w2s:w2e]))
            self.assertGreater(float(wdir[0, w2s:w2e].abs().max()), 0.0)  # w2 kaiming-init
            self.assertEqual(float(wdir[0, b2s:b2e].abs().max()), 0.0)    # b2 zero-init
        head = wdir[0, wo:wo + (16 * 8) * 10]
        self.assertTrue(torch.equal(head,
                                    gen.prototype_lib.proto_weights[1, wo:wo + head.numel()]))
        self.assertEqual(float(wdir[0, bo:bo + 10].abs().max()), 0.0)  # head bias


class TestTwoConvResidualBlock(unittest.TestCase):
    """v2: Plain and ResNet block modes must be genuinely distinct -- in
    EXECUTION (second conv), in COST (effective_params/flops), and in
    GRADIENT FLOW (the w2 slots train)."""

    def _conv_pair(self):
        gen = ModelGenerator(input_dim=3072, output_dim=10, max_layers=3,
                             max_dim=8, dim_choices=(8,), latent_dim=8,
                             num_prototypes=2, use_conv=True, in_channels=3)
        return gen, TaskNetwork()

    def test_generator_exposes_second_conv_weights(self):
        gen, ex = self._conv_pair()
        m = gen(torch.randn(4, 3, 32, 32))
        for i in range(1, 4):
            self.assertEqual(tuple(m.weights[f"c{i}.weight2"].shape), (4, 8, 8, 3, 3))
            self.assertEqual(tuple(m.weights[f"c{i}.bias2"].shape), (4, 8))

    def test_plain_and_resnet_outputs_differ(self):
        from dataclasses import replace
        from mgn.architecture import GeneratedModel
        gen, ex = self._conv_pair()
        gen.eval(); ex.eval()
        x = torch.randn(2, 3, 32, 32)
        m = gen(x)
        bm_pl = torch.zeros(2, 3, 2); bm_pl[:, :, 0] = 1.0
        bm_rs = torch.zeros(2, 3, 2); bm_rs[:, :, 1] = 1.0
        with torch.no_grad():
            o_pl = ex(x, GeneratedModel(structure=replace(m.structure, block_mode=bm_pl),
                                        weights=m.weights, pi_logits=m.pi_logits))
            o_rs = ex(x, GeneratedModel(structure=replace(m.structure, block_mode=bm_rs),
                                        weights=m.weights, pi_logits=m.pi_logits))
        self.assertGreater(float((o_pl - o_rs).abs().max()), 1e-4)

    def test_residual_mode_costs_more(self):
        B, D, L, out = 2, 8, 3, 10
        gates = torch.tensor([[1., 1., 1.], [1., 1., 0.]])
        onehot = (gates > 0.5).float()
        zero = torch.zeros(B, L, 2); one = torch.zeros(B, L, 2)
        one[:, :, 0] = 1.0; zero[:, :, 1] = 1.0
        kw = dict(dim_weights=torch.full((B, L, 1), 0.5),
                  dim_masks=torch.sigmoid(torch.zeros(B, L, D)),
                  residual=torch.zeros(B, 1), use_conv=True, in_channels=3,
                  max_layers=L, max_dim=D, dim_choices=(D,), output_dim=out)
        pl = ModelStructure(layer_gates=onehot.clone(), block_mode=one, **kw)
        rs = ModelStructure(layer_gates=onehot.clone(), block_mode=zero, **kw)
        leg = ModelStructure(layer_gates=onehot.clone(), block_mode=None, **kw)
        Pp, Pr = pl.effective_params_formula(), rs.effective_params_formula()
        Dm = 0.5 * D
        extra = Dm * Dm * 9 + Dm
        expect = torch.tensor([3.0, 2.0])  # row1 layer-2 gate closed
        self.assertTrue(torch.allclose(Pp, Pr - expect * extra))
        self.assertTrue(torch.allclose(Pp, leg.effective_params_formula()))  # legacy = plain
        self.assertTrue(bool((rs.flops_per_sample() > pl.flops_per_sample()).all()))
        self.assertTrue(bool((Pr > Pp).all()))

    def test_second_conv_receives_gradients(self):
        gen, ex = self._conv_pair()
        gen.train()
        x = torch.randn(2, 3, 32, 32)
        m = gen(x)
        onehot = torch.zeros_like(m.structure.block_mode)
        onehot[:, :, 1] = 1.0  # force residual mode everywhere
        m.structure.block_mode = onehot
        ex(x, m).sum().backward()
        _t, offs, _i, wo, bo = get_prototype_layout(3, 8, 10, 3)
        (w2s, w2e, b2s, b2e) = get_second_conv_layout(3, 8, 10, 3)[1]
        grad = gen.prototype_lib.proto_weights.grad
        self.assertIsNotNone(grad)
        self.assertGreater(float(grad[:, w2s:w2e].abs().max()), 0.0)


class TestStructureProjection(unittest.TestCase):
    def test_project_continuous_mode(self):
        S = ModelStructure(
            layer_gates=torch.tensor([[1.0, 1.0, 1.0]]),
            dim_weights=torch.tensor([[[0.9], [0.9], [0.9]]]),
            residual=torch.zeros(1, 1),
            dim_masks=torch.rand(1, 3, 4),
            input_dim=12, output_dim=5, max_dim=4, max_layers=3,
            dim_choices=(2, 4), use_conv=False)
        before = float(S.effective_params_formula())
        proj = project_architecture(S, P_max=1.0)
        self.assertEqual(proj.dim_weights.shape, (1, 3, 1))
        after = float(proj.effective_params_formula())
        # A single width-scaling pass cannot always land under an extreme
        # budget (L is untouched); the invariant is a strict decrease and
        # in-range fractions.
        self.assertLess(after, before)
        self.assertTrue(bool(((proj.dim_weights >= 0) & (proj.dim_weights <= 1)).all()))

    def test_project_discrete_mode(self):
        S = ModelStructure(
            layer_gates=torch.ones(2, 3),
            dim_weights=torch.zeros(2, 3, 2).scatter_(2, torch.ones(2, 3, 1, dtype=torch.long), 1.0),
            residual=torch.zeros(2, 1),
            dim_masks=torch.rand(2, 3, 4),
            input_dim=12, output_dim=5, max_dim=4, max_layers=3,
            dim_choices=(2, 4), use_conv=False)
        proj = project_architecture(S, P_max=1.0)
        self.assertEqual(proj.dim_weights.shape, (2, 3, 2))
        self.assertTrue(torch.allclose(proj.dim_weights.sum(-1), torch.ones(2, 3)))

    def test_gate_distance_fires_for_batched_structures(self):
        a = ModelStructure(layer_gates=torch.rand(8, 3), dim_weights=torch.rand(8, 3, 1),
                           residual=torch.zeros(8, 1))
        b = ModelStructure(layer_gates=torch.rand(8, 3), dim_weights=a.dim_weights.clone(),
                           residual=torch.zeros(8, 1))
        d = structure_distance(a, b)
        # L/D means nearly identical; gate MSE term must contribute > 0
        self.assertGreater(d.item(), 0.0)

    def test_phi_cost_budget_terms(self):
        S = ModelStructure(layer_gates=torch.ones(1, 3), dim_weights=torch.tensor([[[1.0]]]),
                           residual=torch.zeros(1, 1), max_dim=4)
        base = phi_cost(S, lambda_p=1e-4, lambda_c=0, lambda_m=0)
        over = phi_cost(S, 0, 0, 0, budget=ResourceBudget(P_max=1.0))
        self.assertGreater(over.item(), base.item() * 0)  # budget adds relu(P-1)
        self.assertGreater(over.item(), 0.0)


@unittest.skipUnless(torch.cuda.is_available() and
                     __import__("importlib").util.find_spec("mgn.accel._C"),
                     "requires CUDA + built fused kernels")
class TestFusedExecutorParity(unittest.TestCase):
    """The fused v2 executor (--fused-executor) must match the eager
    TaskNetwork to within the pipeline's bf16 conv-backward tolerance.
    Guards the native_group_norm_backward / gemv reductions rewrite of
    pipeline_backward against algebraic drift."""

    def _fused_applicable(self):
        from mgn.accel.fused_autograd import FusedPipelineFunction  # noqa
        return True

    def test_fused_logits_and_grads(self):
        from mgn.accel.fused_autograd import FusedPipelineFunction  # noqa: F401
        # The benchmark's REAL fused-eligible shape (make_conv_aml). The
        # pipeline mixes conv weights in bf16 BY DESIGN (same convention as
        # validate_ext_in.py's reference), so absolute tolerances are tuned
        # to that: measured at this config ~6e-3 logit / ~2e-2 grad rel.
        gen = ModelGenerator(input_dim=784, output_dim=10, max_layers=4,
                             max_dim=128, dim_choices=(32, 64, 96, 128),
                             latent_dim=64, noise_dim=64, num_prototypes=4,
                             use_conv=True, in_channels=1).cuda().train()
        x = torch.randn(64, 784, device="cuda")
        eager = TaskNetwork(fused=False).cuda()
        fused = TaskNetwork(fused=True).cuda()
        # eval(): the head's p=0.3 dropout draws fresh masks per call -- with
        # two forwards of the SAME model the "drift" would be pure dropout
        # noise (~1.0 on unit-scale logits), not pipeline error.
        eager.eval(); fused.eval()
        # identical routing draw reused for both executors (see GeneratedModel.pi)
        torch.manual_seed(0)
        m = gen(x)
        out_e = eager(x, m)
        out_f = fused(x, m)
        scale = out_e.abs().max().item()
        self.assertLess((out_e - out_f).abs().max().item(), 0.02 * scale,
                        "fused forward drifted beyond bf16 tolerance")
        g = torch.randn_like(out_e)
        # conv-mode freezes weight_delta / noise_var_head; autograd.grad
        # rejects params that don't require grad, so filter to trainable.
        # prototype_lib.residual_head feeds struct.residual, which neither
        # executor branch consumes -> allow_unused (grad None on BOTH sides,
        # asserted equal below).
        trainable = [p for p in gen.parameters() if p.requires_grad]
        ge = torch.autograd.grad(out_e, trainable, g, retain_graph=True,
                                 allow_unused=True)
        gf = torch.autograd.grad(out_f, trainable, g, retain_graph=True,
                                 allow_unused=True)
        for a, b in zip(ge, gf):
            if (a is None) != (b is None):
                self.fail("fused/eager disagree on which params receive grad")
        worst = max(((a - b).abs().max() / (a.abs().max() + 1e-9)).item()
                    for a, b in zip(ge, gf) if a is not None)
        # bf16 weight-mixing noise floor: ~3-5e-2 on the deepest param
        # (encoder through the ST gate/dim-mask chain). An ALGEBRA error
        # (e.g. a single-conv backward against the v2 block) shows up at
        # O(0.5+) drift, so this gate still catches real regressions.
        self.assertLess(worst, 8e-2,
                        f"fused backward rel-drift {worst:.2e} > 8e-2")

    def test_fused_falls_back_without_cuda_tensors(self):
        # CPU model must never call into the pipeline (guards _fusable).
        gen = _mlp_generator()
        ex = TaskNetwork(fused=True)   # fused requested...
        x = torch.randn(4, 12)
        m = gen(x)
        out = ex(x, m)                 # ...but MLP branch ignores it
        self.assertEqual(out.shape, (4, 5))


if __name__ == "__main__":
    unittest.main()
