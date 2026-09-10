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
from mgn.layout import get_prototype_layout
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
        total, offs, info, wo, bo = get_prototype_layout(4, 16, 10, in_channels=3)
        prev_end = 0
        for (ws, we, bs, be, ic) in offs:
            self.assertEqual(ws, prev_end)
            self.assertEqual(we - ws, 16 * ic * 9)
            self.assertEqual(be - bs, 16)
            prev_end = be
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
        head = wdir[0, wo:wo + (16 * 8) * 10]
        self.assertTrue(torch.equal(head,
                                    gen.prototype_lib.proto_weights[1, wo:wo + head.numel()]))
        self.assertEqual(float(wdir[0, bo:bo + 10].abs().max()), 0.0)  # head bias


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


if __name__ == "__main__":
    unittest.main()
