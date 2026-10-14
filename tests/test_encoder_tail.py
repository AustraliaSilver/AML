"""Guards for the encoder tail activation.

A terminal ReLU on the vision encoder makes the domain router structurally
incapable of separating tasks: it forces every latent into the non-negative
orthant, pinning pairwise cosine similarity near 1.
"""
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from mgn.blocks import ConvAML_18Layer_Hybrid


def _model(**kw):
    kw.setdefault("K", 6)
    kw.setdefault("max_layers", 18)
    kw.setdefault("max_dim", 32)
    kw.setdefault("num_classes", 10)
    return ConvAML_18Layer_Hybrid(**kw)


def _router(z_by_domain):
    """Centroid router accuracy from a {name: z} dict."""
    names = list(z_by_domain)
    cents = {n: F.normalize(z_by_domain[n].mean(0, keepdim=True), dim=1) for n in names}
    accs = []
    for i, n in enumerate(names):
        sim = torch.stack([(F.normalize(z_by_domain[n], dim=1) * cents[m]).sum(-1)
                           for m in names], dim=1)
        accs.append((sim.argmax(-1) == i).float().mean().item())
    return accs, sum(accs) / len(accs)


class TestEncoderTailActivation(unittest.TestCase):

    def test_default_has_no_terminal_relu(self):
        """Regression: ReLU here collapsed A_auto to 8% while A_oracle held 98%."""
        m = _model()
        self.assertIsInstance(m.vision_encoder[6], nn.Identity)

    def test_relu_is_still_selectable(self):
        m = _model(encoder_tail_act="relu")
        self.assertIsInstance(m.vision_encoder[6], nn.ReLU)

    def test_relu_makes_latents_non_negative(self):
        m = _model(encoder_tail_act="relu").eval()
        with torch.no_grad():
            z = m.vision_encoder(torch.randn(8, 3, 32, 32))
        self.assertGreaterEqual(z.min().item(), 0.0)

    def test_no_tail_keeps_latents_signed(self):
        m = _model(encoder_tail_act="none").eval()
        with torch.no_grad():
            z = m.vision_encoder(torch.randn(8, 3, 32, 32))
        self.assertLess(z.min().item(), 0.0)

    def test_unknown_tail_is_rejected(self):
        with self.assertRaises(KeyError):
            _model(encoder_tail_act="swish")

    def test_router_separates_better_without_relu(self):
        """The operational property that matters: on data drawn from distinct
        domains, a signed latent gives the centroid router usable cosine margins.
        Uses a fixed linear encoder so the comparison is deterministic."""
        torch.manual_seed(0)
        W = torch.randn(16, 32)          # (in, out)
        domains = {
            "a": torch.randn(400, 16) + torch.tensor([3.0] + [0.0] * 15),
            "b": torch.randn(400, 16) - torch.tensor([3.0] + [0.0] * 15),
        }

        def encode(t, act):
            z = t @ W
            return F.relu(z) if act == "relu" else z

        zs_relu = {k: encode(v, "relu") for k, v in domains.items()}
        zs_none = {k: encode(v, "none") for k, v in domains.items()}

        _, mean_relu = _router(zs_relu)
        _, mean_none = _router(zs_none)
        self.assertGreater(mean_none, mean_relu,
                           f"signed latents {mean_none:.3f} should beat "
                           f"relu latents {mean_relu:.3f}")

    def test_centroid_router_unchanged(self):
        """The router itself is fine; only its input geometry changed. Make sure
        the fix did not silently alter its numerics."""
        import inspect
        from train_continual_aml import TaskCentroidRouter
        src = inspect.getsource(TaskCentroidRouter.forward)
        self.assertIn("F.normalize", src)
        self.assertIn("self.temperature", src)


if __name__ == "__main__":
    unittest.main()