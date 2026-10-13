import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mgn.blocks import ConvAML_18Layer_Hybrid
from train_continual_aml import ProtoIsolator


def test_stage_block_mode_hierarchy():
    """Verify that ConvAML_18Layer_Hybrid initializes with the proper 3-stage hierarchy:
    Stage 1: CNN & ResNet dominant, ViT suppressed (no downsampling blur).
    Stage 2: ResNet dominant.
    Stage 3: ViT dominant.
    """
    model = ConvAML_18Layer_Hybrid(K=12, max_layers=18, max_dim=48, num_classes=10)
    bm = model.block_mode_prototypes.data  # [K, 18, 3]

    # Stage 1 (Layers 0..5)
    assert (bm[:, 0:6, 1] > bm[:, 0:6, 0]).all(), "ResNet should dominate Plain CNN in Stage 1"
    assert (bm[:, 0:6, 2] < -1.0).all(), "ViT should be heavily suppressed in Stage 1"

    # Stage 2 (Layers 6..11)
    assert (bm[:, 6:12, 1] > bm[:, 6:12, 0]).all(), "ResNet should dominate Stage 2"
    assert (bm[:, 6:12, 2] < 0.0).all(), "ViT should be suppressed in Stage 2"

    # Stage 3 (Layers 12..17)
    assert (bm[:, 12:18, 2] > bm[:, 12:18, 0]).all(), "ViT should dominate Stage 3"
    assert (bm[:, 12:18, 2] > bm[:, 12:18, 1]).all(), "ViT should dominate ResNet in Stage 3"


def test_classifier_head_dimension_and_norm():
    """Verify 4x4 spatial pooling (768-d) and LayerNorm integration in classifier head."""
    model = ConvAML_18Layer_Hybrid(K=6, max_layers=18, max_dim=48, num_classes=10)
    assert model.head_dim == 48 * 16  # 768
    assert isinstance(model.ln, nn.LayerNorm)
    assert model.ln.normalized_shape == (768,)
    assert model.wo_size == 10 * 768
    assert model.bo_size == 10
    assert model.proto_weights.shape == (6, 10 * 768 + 10)


def test_proto_isolator_zero_drift_assertion():
    """Verify that ProtoIsolator guarantees bitwise zero-drift on frozen prototype rows
    across optimizer steps."""
    model = ConvAML_18Layer_Hybrid(K=6, max_layers=18, max_dim=32, num_classes=10)
    isolator = ProtoIsolator(model)

    # Task 0: owns [0:2)
    s0, e0 = isolator.begin_task(0, 3, reinit=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x0 = torch.randn(4, 3, 32, 32)
    y0 = torch.tensor([0, 1, 2, 3])
    out0, _, _, _ = model(x0)
    loss0 = F.cross_entropy(out0, y0)
    loss0.backward()
    isolator.mask_grad()
    opt.step()
    isolator.restore()

    # Save snapshot of Task 0 rows
    frozen_task0_weights = model.proto_weights.data[:2].clone()
    frozen_task0_bm = model.block_mode_prototypes.data[:2].clone()

    # Task 1: owns [2:4), rows [0:2) are frozen
    s1, e1 = isolator.begin_task(1, 3, reinit=True)
    assert s1 == 2 and e1 == 4
    assert isolator.frozen_hi == 2

    # Multiple aggressive optimizer steps on Task 1
    for _ in range(5):
        opt.zero_grad()
        x1 = torch.randn(4, 3, 32, 32)
        y1 = torch.tensor([4, 5, 6, 7])
        out1, _, _, _ = model(x1)
        loss1 = F.cross_entropy(out1, y1)
        loss1.backward()
        isolator.mask_grad()
        opt.step()
        isolator.restore()

    # Verify ZERO DRIFT on Task 0 prototype weights
    assert torch.equal(model.proto_weights.data[:2], frozen_task0_weights), (
        "Task 0 prototype weights experienced drift!"
    )
    # Verify ZERO DRIFT on Task 0 block mode prototypes
    assert torch.equal(model.block_mode_prototypes.data[:2], frozen_task0_bm), (
        "Task 0 block mode prototypes experienced drift!"
    )


def test_hybrid_18_checkpoint_save_and_load(tmp_path):
    """Verify full state_dict serialization and deserialization integrity."""
    model1 = ConvAML_18Layer_Hybrid(K=6, max_layers=18, max_dim=32, num_classes=10)
    model1.eval()
    x = torch.randn(2, 3, 32, 32)

    with torch.no_grad():
        out1, struct1, pi1, _ = model1(x, tau=0.1)

    ckpt_path = tmp_path / "hybrid_18_test.pth"
    torch.save(model1.state_dict(), ckpt_path)

    model2 = ConvAML_18Layer_Hybrid(K=6, max_layers=18, max_dim=32, num_classes=10)
    model2.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model2.eval()

    with torch.no_grad():
        out2, struct2, pi2, _ = model2(x, tau=0.1)

    assert torch.allclose(out1, out2, atol=1e-5), "Outputs mismatch after loading checkpoint"
    assert torch.allclose(struct1.layer_gates, struct2.layer_gates, atol=1e-5)
    assert torch.allclose(struct1.block_mode, struct2.block_mode, atol=1e-5)
    assert torch.allclose(pi1, pi2, atol=1e-5)


def test_hybrid_18_eval_determinism():
    """Verify that evaluation mode produces completely deterministic predictions."""
    model = ConvAML_18Layer_Hybrid(K=4, max_layers=18, max_dim=32, num_classes=10)
    model.eval()
    x = torch.randn(4, 3, 32, 32)

    with torch.no_grad():
        out_a, struct_a, pi_a, _ = model(x, tau=0.1)
        out_b, struct_b, pi_b, _ = model(x, tau=0.1)

    assert torch.equal(out_a, out_b), "Eval outputs must be bitwise deterministic"
    assert torch.equal(struct_a.block_mode, struct_b.block_mode)
    assert torch.equal(pi_a, pi_b)
