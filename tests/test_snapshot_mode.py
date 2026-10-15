"""Guards for `--snapshot-mode`: what is stored per task, and what is not.

The default mode must keep producing exactly the keys the published protocol
stored before the flag existed, now that the construction was lifted out of the
training loop into `build_task_snapshot`. Everything else here pins the two
cheaper modes, whose whole point is the *absence* of keys -- a regression that
silently re-added the trunk copy would otherwise be invisible.

unittest.TestCase, not bare pytest functions: CI runs
`python -m unittest discover tests` and does not collect pytest-style functions.
"""
import unittest

import torch
import torch.nn as nn

from train_continual_aml import build_task_snapshot, snapshot_bytes


class _Stub(nn.Module):
    """Just enough of the hybrid to exercise every branch of the builder."""

    def __init__(self, trunk=True, struct=True, delta=False):
        super().__init__()
        self.vision_encoder = nn.Linear(4, 4)
        self.prototype_selector = nn.Linear(4, 4)
        self.proto_weights = nn.Parameter(torch.randn(6, 4))
        if trunk:
            self.stem = nn.Conv2d(1, 2, 3, padding=1)
            self.layers = nn.Sequential(nn.Conv2d(2, 2, 3, padding=1))
            self.ln = nn.LayerNorm([2, 1, 1])
        if struct:
            self.structure_prototypes = nn.Parameter(torch.randn(3, 4))
            self.block_mode_prototypes = nn.Parameter(torch.randn(3, 4))
        if delta:
            self.structure_delta = nn.Linear(4, 4)
            self.block_mode_head = nn.Linear(4, 3)


class TestSnapshotMode(unittest.TestCase):
    def test_default_mode_keeps_every_key(self):
        """`full` is the published protocol: nothing may disappear from it."""
        snap = build_task_snapshot(_Stub())
        self.assertEqual(
            set(snap), {"enc", "sel", "stem", "layers", "ln", "struct_heads", "protos"})

    def test_structural_stores_no_weights(self):
        """The point of the mode: only the structure library survives."""
        snap = build_task_snapshot(_Stub(), shared_backbone=True, structural=True)
        self.assertEqual(set(snap), {"struct_heads"})
        # and it must be small -- a few hundred bytes, not megabytes
        self.assertLess(snapshot_bytes(snap), 4096)

    def test_structural_ignores_snap_q8(self):
        """int8 prototypes are still per-task weights, so they must be dropped."""
        snap = build_task_snapshot(_Stub(), shared_backbone=True, structural=True,
                                   snap_q8=True)
        self.assertNotIn("protos", snap)

    def test_shared_mode_drops_the_trunk_but_keeps_the_rest(self):
        """3A: the 81 % trunk copy goes, the encoder and prototypes stay."""
        snap = build_task_snapshot(_Stub(), shared_backbone=True)
        self.assertEqual(set(snap), {"enc", "sel", "struct_heads", "protos"})
        self.assertNotIn("layers", snap)
        self.assertNotIn("stem", snap)

    def test_structural_is_exactly_the_structure_library(self):
        """Not merely "smaller": exactly the structure heads and nothing else.

        The orders-of-magnitude claim (7.209 MB -> 0.046 MB, i.e. ~157x) is a
        property of the real hybrid and is measured, not asserted here -- a stub
        this small cannot exhibit it.
        """
        stub = _Stub()
        structural = build_task_snapshot(stub, shared_backbone=True, structural=True)
        full = build_task_snapshot(stub)
        self.assertEqual(snapshot_bytes(structural),
                         snapshot_bytes({"struct_heads": structural["struct_heads"]}))
        self.assertLess(snapshot_bytes(structural), snapshot_bytes(full))

    def test_snap_q8_stores_int8_codes(self):
        snap = build_task_snapshot(_Stub(), snap_q8=True)
        self.assertIsInstance(snap["protos"], dict)
        self.assertEqual(snap["protos"]["q"].dtype, torch.int8)
        self.assertEqual(snap["protos"]["q"].numel(), 24)

    def test_snap_q8_fp32_path_is_float32(self):
        snap = build_task_snapshot(_Stub(), snap_q8=False)
        self.assertIsInstance(snap["protos"], torch.Tensor)
        self.assertEqual(snap["protos"].dtype, torch.float32)

    def test_values_are_clones_not_live_references(self):
        """A snapshot that aliases the live weights would drift with training."""
        model = _Stub()
        snap = build_task_snapshot(model)
        with torch.no_grad():
            model.vision_encoder.weight.add_(1.0)
            model.proto_weights.add_(1.0)
        self.assertFalse(torch.allclose(snap["enc"]["weight"],
                                        model.vision_encoder.weight))
        self.assertFalse(torch.allclose(snap["protos"], model.proto_weights))

    def test_legacy_delta_model_still_gets_structure_heads(self):
        snap = build_task_snapshot(_Stub(struct=False, delta=True))
        self.assertIn("struct_heads", snap)
        self.assertEqual(set(snap["struct_heads"]), {"delta", "block"})

    def test_model_without_trunk_has_no_trunk_keys(self):
        """ViT-style / trunk-less models must not grow phantom keys."""
        snap = build_task_snapshot(_Stub(trunk=False))
        self.assertEqual(set(snap), {"enc", "sel", "struct_heads", "protos"})


if __name__ == "__main__":
    unittest.main()
