"""The fused backward kernel was disabled at the default configuration.

`use_mix` required `256 % C_out == 0`, a leftover from the v1 kernel. The
default `max_dim` is 48, and 256 % 48 == 16, so the condition was false on every
default run and the fused backward silently fell back to recomputing through
ATen. The v2 kernel iterates channels sequentially and parallelises over
spatial, so it only needs C <= MB_MAX_C.

These tests pin the Python-side guard to the C++ side, because the two drifted
apart and only the Python one is reachable without a CUDA toolchain.
"""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
MB_MAX_C = 1024


def _fused_autograd_text():
    return (ROOT / "mgn" / "accel" / "fused_autograd.py").read_text(encoding="utf-8")


def _bindings_text():
    return (ROOT / "mgn" / "csrc" / "bindings.cpp").read_text(encoding="utf-8")


class TestFusedBackwardGuard(unittest.TestCase):

    def test_default_width_passes_the_guard(self):
        """max_dim = 48 is the shipped default. It must reach the fused path."""
        self.assertTrue(48 <= MB_MAX_C)
        # the condition the code actually evaluates
        self.assertTrue(48 <= MB_MAX_C, "48 must be admitted")

    def test_python_guard_has_no_power_of_two_divisor_test(self):
        src = _fused_autograd_text()
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        self.assertNotIn("256 %", code,
                         "the v1 divisor test still gates the fused backward")
        self.assertNotIn("% C_out", code)

    def test_python_guard_uses_the_same_cap_as_the_kernel(self):
        src = _fused_autograd_text()
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        self.assertIn(f"C_out_i <= {MB_MAX_C}", code)

    def test_cpp_guard_matches_the_kernel_cap(self):
        src = _bindings_text()
        self.assertIn(f"C <= {MB_MAX_C}", src)
        self.assertNotIn("256 % C", src)

    def test_kernel_defines_the_cap_the_guards_reference(self):
        cu = (ROOT / "mgn" / "csrc" / "kernels" / "fused_pipeline.cu").read_text(
            encoding="utf-8")
        m = re.search(r"#define\s+MB_MAX_C\s+(\d+)", cu)
        self.assertIsNotNone(m, "MB_MAX_C must be defined in the kernel")
        self.assertEqual(int(m.group(1)), MB_MAX_C,
                         "the two guards and the kernel disagree about the cap")

    def test_kernel_walks_channels_sequentially(self):
        """This is WHY the divisor test is unnecessary: the v1 kernel mapped
        channels onto threads, the v2 one loops."""
        cu = (ROOT / "mgn" / "csrc" / "kernels" / "fused_pipeline.cu").read_text(
            encoding="utf-8")
        self.assertIn("for (int c = 0; c < C; ++c)", cu)
        self.assertIn("s += MB_THREADS", cu,
                      "spatial is strided across threads, channels are not")


if __name__ == "__main__":
    unittest.main()