"""Regression tests for the HtS depth parameterization (stage_blocks)."""

import unittest
import torch

from hts_lib.models.hypernetwork import HtSContinualModel
from hts_lib.models.losses import compute_layer_aware_diversity_loss


class TestHtSDepth(unittest.TestCase):
    def _forward(self, model, b=3):
        x = torch.randn(b, 3, 32, 32)
        model.eval()
        with torch.no_grad():
            logits, alphas = model(x, task="cifar")
        return logits, alphas

    def test_default_is_the_historical_8_block_network(self):
        m = HtSContinualModel(d1=16, d2=32, d3=48, K=4)
        self.assertEqual(m.n_blocks, 8)
        self.assertEqual(list(m.block_weights.numpy().round(1)),
                         [0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0])
        logits, alphas = self._forward(m)
        self.assertEqual(alphas.shape, (3, 8, 4))
        self.assertEqual(logits.shape, (3, 10))

    def test_shallow_and_deep_layouts(self):
        for stage_blocks, expected in (((1, 1, 1), 5), ((3, 1, 2), 8), ((0, 2, 0), 4)):
            m = HtSContinualModel(d1=8, d2=16, d3=24, K=3, stage_blocks=stage_blocks)
            self.assertEqual(m.n_blocks, expected, stage_blocks)
            _logits, alphas = self._forward(m)
            self.assertEqual(tuple(alphas.shape), (3, expected, 3))
            # router head width and block-weight vector track depth
            self.assertEqual(m.task_ctrl[-1].out_features, expected * 3)
            self.assertEqual(m.block_weights.numel(), expected)
            self.assertTrue(bool(torch.isfinite(m.block_weights).all()))
            l_ent, l_use, l_ort = compute_layer_aware_diversity_loss(
                alphas, m)   # K=None + block-weight derivation
            self.assertTrue(torch.isfinite(l_ent + l_use + l_ort))

    def test_forward_backward_full_train_step(self):
        m = HtSContinualModel(d1=8, d2=16, d3=24, K=3, stage_blocks=(1, 2, 1),
                              num_classes=5)
        m.train()
        x = torch.randn(2, 3, 32, 32)
        y = torch.tensor([0, 4])
        logits, alphas = m(x, task="cifar")
        loss = torch.nn.functional.cross_entropy(logits, y)
        l_ent, l_use, l_ort = compute_layer_aware_diversity_loss(alphas, m)
        (loss + 0.01 * l_ent + 0.1 * l_use + 0.05 * l_ort).backward()
        for name in ("b2_priv", "b3_priv"):   # all three stages must be trained
            self.assertIsNotNone(getattr(m.basis_sys, name).grad)
        stats = m.count_parameters_breakdown(task="cifar")
        self.assertGreater(stats["total_hypernetwork_params"], stats["child_active_params"])


if __name__ == "__main__":
    unittest.main()
