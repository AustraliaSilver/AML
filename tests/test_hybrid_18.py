import torch
import torch.nn.functional as F
from mgn.blocks import ConvAML_18Layer_Hybrid


def test_hybrid_18_initialization():
    model = ConvAML_18Layer_Hybrid(K=12, max_layers=18, max_dim=48, num_classes=10)
    assert len(model.layers) == 18
    assert model.block_mode_prototypes.shape == (12, 18, 3)
    assert model.structure_prototypes.shape == (12, 18 + 18 * 48)
    assert model.proto_weights.shape[0] == 12


def test_hybrid_18_forward_shapes():
    model = ConvAML_18Layer_Hybrid(K=8, max_layers=18, max_dim=32, num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    out, struct, pi, diag = model(x, tau=1.0)

    assert out.shape == (4, 10)
    assert pi.shape == (4, 8)
    assert struct.layer_gates.shape == (4, 18)
    assert struct.block_mode.shape == (4, 18, 3)
    assert diag["bm_logits"].shape == (4, 18, 3)
    assert diag["layer_soft"].shape == (4, 18)
    assert diag["z"].shape == (4, model.latent_dim)


def test_hybrid_18_backward():
    model = ConvAML_18Layer_Hybrid(K=6, max_layers=18, max_dim=32, num_classes=10)
    x = torch.randn(2, 3, 32, 32)
    y = torch.tensor([1, 5])

    out, _, _, _ = model(x, tau=1.0)
    loss = F.cross_entropy(out, y)
    loss.backward()

    assert model.proto_weights.grad is not None
    assert model.block_mode_prototypes.grad is not None
    assert model.structure_prototypes.grad is not None


def test_hybrid_18_struct_override():
    model = ConvAML_18Layer_Hybrid(K=6, max_layers=18, max_dim=32, num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)

    gates = torch.ones(2, 18)
    # Mode: Force layer 0 to Plain, layer 1 to ResNet, layer 2 to ViT
    modes = torch.zeros(2, 18, 3)
    modes[:, 0, 0] = 1.0  # Plain
    modes[:, 1, 1] = 1.0  # ResNet
    modes[:, 2, 2] = 1.0  # ViT

    out, struct, _, _ = model(x, tau=0.1, struct_override=(gates, modes))
    assert out.shape == (2, 10)
    assert torch.allclose(struct.layer_gates, gates)
    assert torch.allclose(struct.block_mode, modes)


def test_hybrid_18_legacy_2mode_override():
    model = ConvAML_18Layer_Hybrid(K=6, max_layers=18, max_dim=32, num_classes=10)
    model.eval()
    x = torch.randn(2, 3, 32, 32)

    gates = torch.ones(2, 18)
    modes_2d = torch.zeros(2, 18, 2)
    modes_2d[:, :, 1] = 1.0  # ResNet

    out, struct, _, _ = model(x, tau=0.1, struct_override=(gates, modes_2d))
    assert out.shape == (2, 10)
    assert struct.block_mode.shape == (2, 18, 3)
    assert torch.allclose(struct.block_mode[:, :, 1], torch.ones(2, 18))
