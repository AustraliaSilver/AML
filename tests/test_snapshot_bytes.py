"""The per-task snapshot size must be reported honestly.

The README's headline storage claim is "+7.5 MB per task". The log line that
produced it counted only the int8 prototype matrix, which in the 18-layer
hybrid is 92 k elements (~90 KB) and therefore printed as "0 MB" - a number that
reads like "nothing is stored" while the snapshot actually holds the fp32
encoder and trunk. `snapshot_bytes` walks the whole snapshot, including the
quantisation scale, so the logged number and the documented number are the same
quantity.
"""
import unittest

import torch

from train_continual_aml import snapshot_bytes


class TestSnapshotBytes(unittest.TestCase):

    def test_counts_tensors_dicts_and_the_quantised_scale(self):
        snap = {
            "enc": {"w": torch.zeros(100, dtype=torch.float32)},      # 400 B
            "protos": {"q": torch.zeros(1000, dtype=torch.int8),      # 1000 B
                       "scale": torch.tensor(0.5)},                   # 4 B
        }
        self.assertEqual(snapshot_bytes(snap), 400 + 1000 + 4)

    def test_dtype_matters(self):
        f32 = {"w": torch.zeros(256, dtype=torch.float32)}
        f16 = {"w": torch.zeros(256, dtype=torch.float16)}
        self.assertEqual(snapshot_bytes(f32), 1024)
        self.assertEqual(snapshot_bytes(f16), 512)

    def test_ignores_non_tensor_scalars(self):
        snap = {"tag": "mnist", "gate": torch.zeros(4)}
        self.assertEqual(snapshot_bytes(snap), 16)


if __name__ == "__main__":
    unittest.main()