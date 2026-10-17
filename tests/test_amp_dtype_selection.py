"""The autocast dtype must be chosen by bf16 TENSOR CORES, not by bf16 support.

Measured on a Tesla T4 (sm_75), 18-layer AML, batch 256, 32x32, norm=in:

    fp32      314.1 ms
    bf16     1072.4 ms
    fp16      266.8 ms

so bf16 on a pre-Ampere GPU is 3.4x slower than not using tensor cores at all.
The trap is that `torch.cuda.is_bf16_supported()` returns **True** on that T4:
basic bf16 ops work, they are just emulated, with no bf16 tensor core behind
them. The old gate therefore took the bf16 branch on exactly the hardware where
it is worst.

unittest.TestCase, not bare pytest functions: CI also runs
`python -m unittest discover tests`.
"""

import unittest
from unittest import mock

import torch

from train_continual_aml import resolve_amp_dtype


# (major, minor) -> expected dtype. sm_80 is where the bf16 `mma.sync`
# instructions arrive, so it is the boundary in both directions.
CASES = [
    ((7, 0), torch.float16, "Volta V100: fp16 tensor cores only"),
    ((7, 5), torch.float16, "Turing T4: fp16 tensor cores, no bf16 ones"),
    ((8, 0), torch.bfloat16, "Ampere A100: first device with bf16 tensor cores"),
    ((8, 6), torch.bfloat16, "consumer Ampere RTX 30-series"),
    ((8, 9), torch.bfloat16, "Ada"),
    ((9, 0), torch.bfloat16, "Hopper"),
]


class TestResolveAmpDtype(unittest.TestCase):

    def _resolve(self, cap, name="fake"):
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=cap), \
             mock.patch.object(torch.cuda, "get_device_name", return_value=name):
            return resolve_amp_dtype(torch.device("cuda"))

    def test_dtype_follows_bf16_tensor_core_availability(self):
        for cap, want, why in CASES:
            with self.subTest(cap=cap, why=why):
                got, _reason = self._resolve(cap)
                self.assertIs(got, want, f"{cap}: {why}")

    def test_reason_names_the_device_and_the_reason(self):
        _, reason = self._resolve((7, 5), name="Tesla T4")
        self.assertIn("Tesla T4", reason)
        self.assertIn("sm_75", reason)
        self.assertIn("sm_80", reason,
                      "the fallback has to say where bf16 support starts")

    def test_capable_device_says_so(self):
        _, reason = self._resolve((8, 6), name="NVIDIA GeForce RTX 3050")
        self.assertIn("bf16 tensor cores", reason)
        self.assertNotIn("no bf16", reason)

    def test_cpu_disables_autocast(self):
        got, reason = resolve_amp_dtype(torch.device("cpu"))
        self.assertIsNone(got)
        self.assertIn("cpu", reason)

    def test_t4_style_device_contradicts_torchs_own_helper(self):
        """The whole bug in one test: the hardware PyTorch calls 'bf16 supported'
        is the hardware we must NOT use bf16 on."""
        with mock.patch.object(torch.cuda, "is_available", return_value=True), \
             mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)), \
             mock.patch.object(torch.cuda, "get_device_name", return_value="Tesla T4"), \
             mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            got, _ = resolve_amp_dtype(torch.device("cuda"))
            self.assertTrue(torch.cuda.is_bf16_supported(),
                            "precondition: PyTorch claims support here")
            self.assertIs(got, torch.float16,
                          "but bf16 must still be refused: no tensor cores")


if __name__ == "__main__":
    unittest.main()
