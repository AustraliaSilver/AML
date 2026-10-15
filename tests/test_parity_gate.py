"""The CUDA parity gate must be noise-relative, not a hand-picked constant.

`validate_ext_in.py` compares a bf16 fused kernel against a bf16 eager
reference. The reference disagrees with itself by 6-28% depending on depth
(B39 in the audit log), so any fixed tolerance is either unsatisfiable or
meaningless. The gate therefore compares against the reference's own measured
spread.

This test is source-level on purpose: the measurement needs a GPU, but the
*rule* must hold on a CPU-only CI runner too, and a regression here is a silent
mis-measurement rather than a crash.
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "validate_ext_in.py"


class TestParityGateIsNoiseRelative(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.text = SRC.read_text(encoding="utf-8")
        cls.code = "\n".join(l for l in cls.text.splitlines()
                             if not l.strip().startswith("#"))

    def test_file_exists(self):
        self.assertTrue(SRC.is_file())

    def test_declares_a_noise_factor_above_one(self):
        m = re.search(r"NOISE_FACTOR\s*=\s*([0-9.]+)", self.code)
        self.assertIsNotNone(m, "NOISE_FACTOR must be declared")
        self.assertGreater(float(m.group(1)), 1.0,
                           "a factor of 1.0 would fail a kernel that merely "
                           "matches the reference's own rounding")

    def test_no_bare_fixed_tolerance_comparisons_remain(self):
        """`err < FWD_TOL` / `worst < GRAD_TOL` are exactly the unsatisfiable
        gates B39 removed."""
        for pat in (r"<\s*FWD_TOL\b", r"<\s*GRAD_TOL\b"):
            self.assertIsNone(re.search(pat, self.code),
                              f"bare fixed-tolerance comparison {pat!r} is back")

    def test_tolerances_are_computed_from_a_measured_floor(self):
        self.assertIn("ref_noise", self.code)
        self.assertIn("grad_noise", self.code)
        self.assertIn("NOISE_FACTOR *", self.code)

    def test_noise_is_measured_as_bf16_vs_fp32(self):
        """A floor measured any other way would not bound the bf16 rounding."""
        self.assertRegex(self.code, r"dtype=torch\.float32")
        self.assertRegex(self.code, r"dtype=torch\.bfloat16")

    def test_every_gated_result_is_reported_with_its_floor(self):
        self.assertIn("reference noise", self.code)
        self.assertIn("vs tol", self.code)


if __name__ == "__main__":
    unittest.main()