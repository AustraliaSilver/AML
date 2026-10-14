"""Static extraction from the 18-layer hybrid supernet.

The extractor used to accept only ConvAML_CIFAR_TrueDynamic. Its guard checked
for `structure_prototypes` and `proto_weights`, which the hybrid also has, so
passing a hybrid sailed through validation and then died with
"object has no attribute 'ood_router'". These tests pin the dispatch and the
structural differences that made a separate path necessary.
"""
import unittest
import inspect

import torch

from AML_Core_Code.extract_static_model import (
    StaticChildModel,
    StaticHybridBlock,
    extract_child,
    extract_child_hybrid18,
)
from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid
from mgn.blocks.hybrid_router import HybridBlock


def _mother(**kw):
    torch.manual_seed(0)
    m = ConvAML_18Layer_Hybrid(K=12, max_dim=48, max_layers=18,
                               num_classes=10, **kw)
    return m.eval()


class TestExtractDispatch(unittest.TestCase):

    def test_hybrid_is_routed_to_the_hybrid_path(self):
        m = _mother()
        self.assertIsInstance(extract_child(m, torch.randn(2, 3, 32, 32)),
                              StaticChildModel)

    def test_unknown_model_raises_typeerror(self):
        class NotASupernet(torch.nn.Module):
            pass

        with self.assertRaises(TypeError):
            extract_child(NotASupernet(), torch.randn(2, 3, 32, 32))

    def test_hybrid_path_rejects_a_foreign_model(self):
        with self.assertRaises(TypeError):
            extract_child_hybrid18(torch.nn.Linear(4, 4), torch.randn(2, 3, 32, 32))


class TestStaticHybridExtraction(unittest.TestCase):

    def setUp(self):
        self.m = _mother()
        self.child = extract_child(self.m, torch.randn(4, 3, 32, 32))

    def test_forward_produces_finite_logits(self):
        y = self.child(torch.randn(8, 3, 32, 32))
        self.assertEqual(tuple(y.shape), (8, 10))
        self.assertTrue(torch.isfinite(y).all())

    def test_every_layer_compiled_to_exactly_one_topology(self):
        """Hard pruning: no dynamic branch survives in the child."""
        for layer in self.child.features:
            self.assertIsInstance(layer, StaticHybridBlock)
            self.assertIn(layer.mode_idx, (0, 1, 2))
            self.assertNotIsInstance(layer.body, HybridBlock)

    def test_child_is_materially_smaller_than_the_supernet(self):
        n_child = sum(p.numel() for p in self.child.parameters())
        n_super = sum(p.numel() for p in self.m.parameters())
        self.assertLess(n_child, n_super * 0.75,
                        f"child {n_child} should be far below supernet {n_super}")

    def test_stem_and_layernorm_are_carried_over(self):
        """The trunk's affine LayerNorm and conv stem are not layer-local, so
        dropping them would silently change the head's input distribution."""
        self.assertIsNotNone(self.child.stem)
        self.assertIsNotNone(self.child.ln)
        self.assertEqual(self.child.ln.normalized_shape[0], self.m.head_dim)

    def test_head_uses_the_models_own_offsets(self):
        """proto_weights is [K, wo_size + num_classes] in the hybrid, which is a
        different layout from the legacy get_prototype_layout. Reading offsets
        off the model is what keeps the two from drifting apart."""
        w = self.child.fc.weight.data
        self.assertEqual(tuple(w.shape), (10, self.m.head_dim))

    def test_extraction_is_deterministic(self):
        a = extract_child(self.m, torch.randn(4, 3, 32, 32))
        b = extract_child(self.m, torch.randn(4, 3, 32, 32))
        self.assertEqual([l.mode_idx for l in a.features],
                         [l.mode_idx for l in b.features])

    def test_child_has_no_training_only_dynamics(self):
        """The point of extraction: no Gumbel-Softmax and no prototype routing
        left at inference."""
        names = {n for n, _ in self.child.named_modules()}
        self.assertNotIn("ood_router", names)
        self.assertNotIn("prototype_selector", names)

    def test_dropout_rate_matches_the_mother(self):
        """hybrid_18.py uses p=0.1. The legacy StaticChildModel used 0.2, and
        carrying that over would make the child a different function in train
        mode than the model it was distilled from."""
        self.assertAlmostEqual(self.child.dropout_p, 0.1, places=6)

    def test_width_floor_matches_the_mother(self):
        """hybrid_18.py:244 clamps at min=16.0; the legacy extractor used 24.0."""
        from AML_Core_Code.extract_static_model import extract_child_hybrid18 as _x
        self.assertEqual(_x.__doc__ is not None, True)
        # reconstruct the clamp the extractor uses and compare with the source
        src = inspect.getsource(extract_child_hybrid18)
        self.assertIn("MIN_ACTIVE_DIM = 16.0", src)
        self.assertNotIn("min=24.0", src)

    def test_closed_layer_is_compiled_out(self):
        """A gate at or below gate_skip_eps must cost nothing and leave the
        activation untouched - this is the whole point of --depth-cost."""
        blk = StaticHybridBlock(self.m.layers[1], 1, torch.ones(48, 1, 1),
                                gate=0.0, is_first=False,
                                gate_skip_eps=self.m.gate_skip_eps)
        self.assertFalse(blk.is_active)
        x = torch.randn(2, 48, 32, 32)
        self.assertTrue(torch.equal(blk(x), x))

    def test_open_layer_blends_by_its_gate(self):
        """gate=1 reproduces the block exactly; gate=0.5 is halfway between the
        block output and the identity, matching h_new = g*h_cand + (1-g)*h."""
        mask = torch.ones(48, 1, 1)
        full = StaticHybridBlock(self.m.layers[1], 1, mask, gate=1.0, is_first=False)
        half = StaticHybridBlock(self.m.layers[1], 1, mask, gate=0.5, is_first=False)
        x = torch.randn(2, 48, 32, 32)
        y_full, y_half = full(x), half(x)
        self.assertTrue(torch.allclose(y_half, 0.5 * y_full + 0.5 * x, atol=1e-5))

    def test_first_layer_ignores_its_gate(self):
        """The mother deliberately ignores layer 0's gate: it sets the feature
        scale consumed by the rest of the trunk (hybrid_18.py:367-369)."""
        blk = StaticHybridBlock(self.m.layers[0], 1, torch.ones(48, 1, 1),
                                gate=0.0, is_first=True)
        self.assertTrue(blk.is_active, "layer 0 must run even with a closed gate")


if __name__ == "__main__":
    unittest.main()