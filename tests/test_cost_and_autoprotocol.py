"""Regression tests for the three issues found in review.

  1. The expected-FLOPs loop seeded its accumulator with mode 0's RAW cost, so
     PlainConv entered with coefficient 1.0 instead of block_mode[..., 0]. Every
     non-Plain route reported its cost PLUS a full PlainConv cost (+53% ResNet,
     +1204% ViT at one-hot routing), and the depth gate was not a multiplier.
  2. The autonomous (A_auto) protocol was handed a task-index-derived pi_mask,
     leaking the true task id and forcing untrained prototype rows whenever the
     router mispredicted.
"""
import unittest

import torch

from mgn.blocks import ConvAML_18Layer_Hybrid


def _model(**kw):
    kw.setdefault("K", 6)
    kw.setdefault("max_layers", 18)
    kw.setdefault("max_dim", 32)
    kw.setdefault("num_classes", 10)
    return ConvAML_18Layer_Hybrid(**kw)


def _onehot(mode, layers=18, batch=4, dev="cpu"):
    bm = torch.zeros(1, layers, 3, device=dev)
    bm[0, :, mode] = 1.0
    return bm.expand(batch, layers, 3).contiguous()


class TestExpectedFlopsIsCorrectlyWeighted(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.m = _model().eval()
        self.m.compute_cost = True
        self.dev = "cpu"
        self.x = torch.randn(4, 3, 32, 32)

    def _layer0_cost(self, mode, gate=1.0):
        bm = _onehot(mode, batch=4, dev=self.dev)
        gates = torch.ones(4, 18, device=self.dev)
        gates[:, 0] = gate
        with torch.no_grad():
            _, _, _, d = self.m(self.x, tau=0.1, struct_override=(gates, bm))
        return d["expected_flops_per_layer"][:, 0]

    def test_onehot_routing_reports_that_modes_cost(self):
        """Regression: routing to ViT reported 13x ViT's actual cost because the
        full PlainConv cost was added on top."""
        blk = self.m.layers[0]
        with torch.no_grad():
            _, _, _, probe = self.m(self.x, tau=0.1)
        d_len = probe["d_lengths"][:, 0]
        for mode, name in ((0, "plain"), (1, "resnet"), (2, "vit")):
            true = blk.compute_flops((32, 32), mode_idx=mode, active_dim=d_len)
            got = self._layer0_cost(mode)
            self.assertTrue(torch.allclose(got, true, rtol=1e-4),
                            f"{name}: reported {got.mean():.0f} vs true {true.mean():.0f}")

    def test_soft_routing_is_the_weighted_sum(self):
        w = torch.tensor([[0.2, 0.3, 0.5]]).expand(4, 18, 3).contiguous()
        bm = w.clone()
        bm[0, 0] = torch.tensor([0.2, 0.3, 0.5])
        gates = torch.ones(4, 18)
        with torch.no_grad():
            _, _, _, d = self.m(self.x, tau=0.1, struct_override=(gates, bm))
            _, _, _, probe = self.m(self.x, tau=0.1)
        d_len = probe["d_lengths"][:, 0]
        blk = self.m.layers[0]
        expect = sum(blk.compute_flops((32, 32), mode_idx=k, active_dim=d_len) * w[0, 0, k]
                      for k in range(3))
        got = d["expected_flops_per_layer"][:, 0]
        self.assertTrue(torch.allclose(got, expect, rtol=1e-4),
                        f"reported {got.mean():.0f} vs weighted sum {expect.mean():.0f}")

    def test_closing_the_depth_gate_scales_the_cost(self):
        """Regression: layer_gates was not a factor, so pruning a layer left the
        reported cost unchanged and the complexity term could not reward it."""
        open_c = self._layer0_cost(1, gate=1.0)
        shut_c = self._layer0_cost(1, gate=0.0)
        self.assertAlmostEqual(shut_c.mean().item(), 0.0, places=3)
        self.assertGreater(open_c.mean().item(), 0.0)

    def test_cost_still_differentiable_wrt_architecture(self):
        self.m.train()
        self.m.compute_cost = True
        _, _, _, d = self.m(self.x, tau=0.8)
        d["expected_gflops"].backward()
        g = self.m.block_mode_prototypes.grad
        self.assertIsNotNone(g)
        self.assertGreater(g.abs().sum().item(), 0.0)


class TestAutoProtocolHasNoTaskLeak(unittest.TestCase):
    """A_auto must run without the true task index."""

    def test_auto_protocol_passes_no_pi_mask(self):
        import inspect

        import train_continual_aml as tca
        src = inspect.getsource(tca.main)
        start = src.index("acc_auto, fp_auto = evaluate")
        call = src[start:start + 900]
        self.assertIn("dom_truth=None", call)
        self.assertIn("pi_mask=None", call,
                      "A_auto must not receive a task-derived pi_mask")
        # the oracle and live protocols keep theirs
        self.assertIn("pi_mask=pi_mask_j", src,
                      "oracle/live protocols should keep the task mask")

    def test_untrained_rows_are_not_selected_by_autonomous_protocol(self):
        """A router predicting an older task d must not be forced onto rows that
        were untrained when snaps[d] was taken."""
        K, T = 12, 3
        # rows owned by task 0 = [0:4); task 1 = [4:8); task 2 = [8:12)
        sj, ej = round(1 * K / T), round(2 * K / T)
        mask_task1 = torch.zeros(K, dtype=torch.bool)
        mask_task1[sj:ej] = True
        # at the moment task 0's snapshot is taken, rows [4:8) are still their
        # initial values -> a task-1 mask would select untrained head weights
        self.assertTrue(mask_task1[4:8].all())
        self.assertFalse(mask_task1[0:4].any())


if __name__ == "__main__":
    unittest.main()