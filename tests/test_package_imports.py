"""Every shipped module must still import.

The repository has been refactored repeatedly (make_norm, the router, the static
extractor, the fused-kernel guards) and a refactor that removes or renames one
symbol breaks importers that no test happens to touch. `ruff`'s F401/F821 pass
is per-file and never executes an import, so it cannot see this class of break;
CI would go green while `python benchmarks/...` died on the user's machine.

This walks the package tree rather than hard-coding a list, so a new module is
covered the moment it is added. Modules that legitimately need hardware or
network at import time are skipped explicitly rather than by prefix guessing.
"""
import importlib
import pathlib
import pkgutil
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Import-time side effects that a unit test must not trigger.
SKIP_SUFFIXES = ("benchmark_pipeline.py", "benchmark_speed.py",
                 "test_backward.py", "test_fused_conv.py")


def _package_files(pkg_name):
    pkg = importlib.import_module(pkg_name)
    root = pathlib.Path(pkg.__file__).parent
    out = []
    for m in pkgutil.walk_packages([str(root)], prefix=pkg_name + "."):
        if m.name.rsplit(".", 1)[-1] + ".py" in SKIP_SUFFIXES:
            continue
        out.append(m.name)
    return out


class TestPackageImports(unittest.TestCase):

    def _import_all(self, pkg_name):
        names = _package_files(pkg_name)
        self.assertGreater(len(names), 5, "walk found suspiciously few modules")
        for name in names:
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_mgn_imports(self):
        self._import_all("mgn")

    def test_hts_lib_imports(self):
        self._import_all("hts_lib")

    def test_static_extractor_imports(self):
        """AML_Core_Code is not a package (no __init__.py); pytest's
        `pythonpath = [".", "AML_Core_Code"]` puts it on sys.path, which is how
        the paper's reproduction instructions import it."""
        mod = importlib.import_module("extract_static_model")
        self.assertTrue(hasattr(mod, "extract_child_hybrid18"))
        self.assertEqual(pathlib.Path(mod.__file__).parent.name, "AML_Core_Code")


if __name__ == "__main__":
    unittest.main()