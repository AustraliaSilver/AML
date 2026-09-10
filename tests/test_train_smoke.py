"""
CPU-safe smoke tests for the training entry points (regression coverage that
was previously exercised only by long GPU runs or not at all in CI):

* ConvAML_CIFAR_TrueDynamic forward/backward + deterministic eval mode
  (pins the eval-time-Gumbel bug class).
* Depth gates must be per-layer independent (pins the cross-layer gumbel
  winner-take-all bug class).
* ProtoIsolator masked-gradient semantics: frozen rows get zero grad and
  must not drift through the optimizer step + restore.
* train.main() end-to-end plumbing with synthetic data + --steps cap
  (bi-level step, model selection on VAL, checkpoint save).

Run: python -m unittest tests.test_train_smoke
"""

import os
import sys
import tempfile
import types
import unittest

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import train as train_mod  # noqa: E402  (CUDA ext bootstrap is guarded)


class TestSupernetDynamics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = train_mod.ConvAML_CIFAR_TrueDynamic(
            K=4, latent_dim=32, max_layers=3, max_dim=8, norm_mode="gn1")
        self.x = torch.randn(6, 3, 32, 32)
        self.y = torch.randint(0, 10, (6,))

    def test_forward_backward(self):
        self.model.train()
        out, struct, pi, diag = self.model(self.x, tau=1.0)
        self.assertEqual(out.shape, (6, 10))
        self.assertTrue(struct.layer_gates.requires_grad)
        F.cross_entropy(out, self.y).backward()
        self.assertTrue(any(p.grad is not None for p in self.model.parameters()),
                        "no parameter received a gradient")

    def test_eval_is_deterministic(self):
        self.model.eval()
        with torch.no_grad():
            out1, s1, *_ = self.model(self.x, tau=0.1)
            out2, s2, *_ = self.model(self.x, tau=0.1)
        self.assertTrue(torch.equal(out1, out2))
        self.assertTrue(torch.equal(s1.block_mode, s2.block_mode))

    def test_depth_gates_are_not_winner_take_all(self):
        self.model.eval()
        with torch.no_grad():
            _, struct, _, _ = self.model(self.x)
        g = struct.layer_gates
        self.assertEqual(g.shape, (6, 3))
        self.assertTrue(bool(((g >= 0) & (g <= 1)).all()))
        # After the +2.0 warm-up logit init most layers must be open; the old
        # cross-layer normalization forced rowsum ~= 1 per sample.
        self.assertGreater(float(g.sum(-1).mean()), 2.0)


class TestProtoIsolator(unittest.TestCase):
    def test_frozen_rows_no_grad_no_drift(self):
        import train_continual_aml as tca

        class _M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proto_weights = torch.nn.Parameter(torch.randn(8, 5))

        m = _M()
        iso = tca.ProtoIsolator(m)
        s, e = iso.begin_task(2, 4)          # rows [4:6) active, [0:4)+[6:8) frozen
        before = m.proto_weights.detach().clone()
        opt = torch.optim.AdamW([m.proto_weights], lr=1e-1)

        loss = (m.proto_weights * torch.ones_like(m.proto_weights)).sum()
        loss.backward()
        iso.mask_grad()
        grad = m.proto_weights.grad
        self.assertTrue(bool((grad[:s] == 0).all()), "past rows still receive grad")
        self.assertTrue(bool((grad[e:] == 0).all()), "future rows still receive grad")
        self.assertFalse(bool((grad[s:e] == 0).all()), "active rows lost their grad")

        opt.step()
        iso.restore()
        after = m.proto_weights.detach()
        self.assertTrue(torch.equal(after[:s], before[:s]), "past rows drifted")
        self.assertTrue(torch.equal(after[e:], before[e:]), "future rows drifted")
        self.assertFalse(torch.equal(after[s:e], before[s:e]), "active rows froze")


class _ConstTransform:
    def __call__(self, img):
        return img


class _Compose:
    def __init__(self, ops):
        self.transforms = list(ops)

    def __call__(self, img):
        return img


def _make_tv_stub(ds_cls):
    """Torchvision stand-in whose datasets are small random tensor sets, so
    train.main() plumbing can run with zero downloads."""
    tf = types.SimpleNamespace(
        Compose=_Compose,
        ToTensor=_ConstTransform(), Normalize=_ConstTransform(),
        RandomCrop=_ConstTransform(), RandomHorizontalFlip=_ConstTransform(),
        Grayscale=_ConstTransform(), Resize=_ConstTransform(),
    )
    dsets = types.SimpleNamespace(
        MNIST=lambda *a, **k: ds_cls(),
        FashionMNIST=lambda *a, **k: ds_cls(),
        CIFAR10=lambda *a, **k: ds_cls(),
    )
    return types.SimpleNamespace(datasets=dsets, transforms=tf)


class TestTrainMainSmoke(unittest.TestCase):
    def test_steps_capped_run_saves_val_selected_checkpoint(self):
        # Fixed, class-0 labels so an untrained model scores > 0 on val with
        # high probability => the best-val checkpoint path is exercised.
        class _Tiny(torch.utils.data.TensorDataset):
            def __init__(self):
                super().__init__(torch.randn(32, 3, 32, 32),
                                 torch.randint(0, 2, (32,)))

        orig_tv = train_mod.torchvision
        train_mod.torchvision = _make_tv_stub(_Tiny)
        try:
            with tempfile.TemporaryDirectory() as d:
                out_path = os.path.join(d, "best.pth")
                train_mod.main([
                    "--dataset", "mnist", "--epochs", "7", "--steps", "2",
                    "--batch-size", "8", "--K", "4", "--max-dim", "8",
                    "--max-layers", "2", "--workers", "0", "--out", out_path,
                    "--seed", "7", "--data-root", d,
                ])
                # epochs=7 crosses the arch-unfreeze boundary (epoch 5) so the
                # bi-level optimizer_arch step path is also exercised.
                self.assertTrue(os.path.exists(out_path),
                                "main() never saved a best-val checkpoint")
                sd = torch.load(out_path, map_location="cpu", weights_only=True)
                self.assertIn("proto_weights", sd)
        finally:
            train_mod.torchvision = orig_tv


if __name__ == "__main__":
    unittest.main()
