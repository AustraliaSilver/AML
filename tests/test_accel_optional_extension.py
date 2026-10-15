"""`mgn.accel` must import on a machine without the compiled CUDA extension.

The extensions are build artifacts (`.pyd` / `.so`) and are deliberately not
tracked, so every CPU-only checkout - including the GitHub Actions test runner -
has no `mgn.accel._C`. `mgn/executor.py` and `train_cv_cifar_18.py` both wrap
the import in try/except and fall back to the pure-PyTorch path, but the modules
themselves imported the extension at module scope, so a missing *optional*
kernel raised `ModuleNotFoundError: No module named 'mgn.accel._C'` from inside
what looks like a perfectly ordinary package.

The blocker below removes the extension from the import system rather than
moving files on disk, so the test runs identically on a CUDA developer machine
and on a CPU-only CI runner - which is the whole point: it must hold on both.
"""
import importlib
import subprocess
import sys
import textwrap
import unittest

BLOCKER = textwrap.dedent('''
    import sys

    BLOCKED = {"mgn.accel._C", "mgn.accel._Cin"}

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname in BLOCKED:
                raise ImportError("no extension in this environment")
            return None

    sys.meta_path.insert(0, _Blocker())

    import torch
    import mgn.accel.fused_autograd as ga
    import mgn.accel.fused_autograd_in as gai

    assert ga._C is None, "the extension import must have failed"
    assert gai._Cin is None, "the extension import must have failed"

    # The guards must fire with a message that names the fix, not a bare
    # AttributeError on None.
    for fn in (ga._require_C, gai._require_Cin):
        try:
            fn()
        except ImportError as e:
            assert "build_ext.py" in str(e), str(e)
        else:
            raise AssertionError("guard did not fire")

    # The ATen fallback must still produce a gradient-shaped result.
    t = torch.randn(2, 4, 8, 8)
    g = torch.randn_like(t)
    for mode in ("gn1", "in"):
        groups = ga._norm_groups(4, mode)
        mean = torch.zeros(2, groups)
        rstd = torch.ones(2, groups)
        out = ga._norm_bwd(g, t, mean, rstd, mode)
        assert tuple(out.shape) == tuple(t.shape), (mode, out.shape)

    print("OK")
''')


class TestAccelExtensionIsOptional(unittest.TestCase):

    def test_modules_import_and_fail_only_when_used(self):
        # A subprocess, because sys.meta_path is process-global and poisoning it
        # in-process would leak into every other test in the session.
        r = subprocess.run([sys.executable, "-c", BLOCKER],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("OK", r.stdout)

    def test_package_imports_normally_here(self):
        """The guard must not be so eager that it breaks a real CUDA host."""
        ga = importlib.import_module("mgn.accel.fused_autograd")
        self.assertTrue(callable(ga._require_C))


if __name__ == "__main__":
    unittest.main()