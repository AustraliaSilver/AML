"""Packaging invariants for the PyPI release.

Every failure mode here is invisible locally and only shows up on someone
else's machine, after the release is already immutable on PyPI:

* a `py3-none-any` wheel that carries a Windows `.pyd` (imports fine here,
  ModuleNotFoundError everywhere else);
* a subpackage that exists in the checkout but is not shipped, because
  `packages.find` requires an `__init__.py` and someone added a directory
  without one;
* an sdist that cannot build the CUDA kernels because the `.cu` sources were
  never added to `MANIFEST.in`;
* a tag that disagrees with `pyproject.toml` version, so the upload is
  rejected after the tag is already public;
* a publish workflow that quietly relies on a stored API token.

The guards are deliberately text/structural: none of them needs to build a
wheel, so they run in the normal CPU test job.
"""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_pyproject():
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        try:
            import tomli as tomllib
        except ModuleNotFoundError:  # pip vendors a parser on every version
            from pip._vendor import tomli as tomllib
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


PYPROJECT = _load_pyproject()
PROJECT = PYPROJECT["project"]
WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"

# Directories that hold files but are deliberately not importable packages.
# Keep in sync with `exclude` in [tool.setuptools.packages.find].
NON_PACKAGE_DIRS = {"mgn/csrc", "mgn/tests"}


class TestPublishedMetadata(unittest.TestCase):

    def test_name_is_the_project_the_release_guide_uses(self):
        self.assertEqual(PROJECT["name"], "aml-framework")

    def test_version_is_releasable(self):
        version = PROJECT["version"]
        self.assertRegex(version, r"^\d+\.\d+\.\d+$",
                         "PyPI tags are `v<version>` with no local/dev suffix")
        # The publish workflow checks the exact same string against the tag.
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('.lstrip("v")', workflow,
                      "the tag/version check no longer strips the `v` prefix")

    def test_license_metadata_is_the_modern_spdx_form(self):
        """`license = { file = ... }` is deprecated and stops building in Feb
        2027; PEP 639 also forbids pairing it with a License classifier."""
        self.assertEqual(PROJECT["license"], "MIT")
        self.assertEqual(PROJECT["license-files"], ["LICENSE"])
        self.assertNotIn("License ::", "\n".join(PROJECT.get("classifiers", [])))
        self.assertTrue((ROOT / "LICENSE").read_text(encoding="utf-8")
                        .startswith("MIT License"))

    def test_setuptools_is_new_enough_for_the_metadata(self):
        requires = " ".join(PYPROJECT["build-system"]["requires"])
        match = re.search(r"setuptools>=(\d+)", requires)
        self.assertIsNotNone(match, f"no setuptools floor in: {requires}")
        self.assertGreaterEqual(int(match.group(1)), 77,
                                "license-files (PEP 639) needs setuptools>=77")

    def test_python_floor_matches_the_classifiers(self):
        self.assertEqual(PROJECT["requires-python"], ">=3.10")
        for minor in ("3.10", "3.12"):
            self.assertIn(f"Programming Language :: Python :: {minor}",
                          PROJECT["classifiers"])
        self.assertIn("torch>=2.0.0", PROJECT["dependencies"])


class TestWheelContents(unittest.TestCase):

    def test_include_package_data_is_false(self):
        """The single most dangerous flag in this file.

        True makes setuptools copy every file that a MANIFEST/VCS lookup can
        see out of the package directories -- including a locally built
        `mgn/accel/_C.cp312-win_amd64.pyd` -- into a wheel that still claims
        `py3-none-any`. The wheel then installs everywhere and imports nowhere
        except the machine that built it.
        """
        self.assertFalse(PYPROJECT["tool"]["setuptools"]["include-package-data"])

    def test_packages_find_is_an_allow_list(self):
        find = PYPROJECT["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(find["namespaces"], False)
        self.assertEqual(sorted(find["include"]), ["hts_lib*", "mgn*"])
        for excluded in ("mgn.tests*", "mgn.csrc*"):
            self.assertIn(excluded, find["exclude"])

    def test_no_python_directory_is_silently_dropped(self):
        """`namespaces = false` ships only directories with an `__init__.py`.

        A new `mgn/foo/bar.py` without one imports from a checkout and fails
        from the wheel, with nothing in CI to notice. Either add the
        `__init__.py` or list the directory in NON_PACKAGE_DIRS *and* in
        `packages.find.exclude`.
        """
        dropped = []
        for pkg in ("mgn", "hts_lib"):
            for path in (ROOT / pkg).rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                rel_dir = path.parent.relative_to(ROOT).as_posix()
                if rel_dir in NON_PACKAGE_DIRS or rel_dir == pkg:
                    continue
                if not (path.parent / "__init__.py").is_file():
                    dropped.append(f"{rel_dir}/{path.name}")
        self.assertEqual(dropped, [],
                         "directories with .py files but no __init__.py are "
                         "not shipped in the wheel")


class TestSdistContents(unittest.TestCase):

    def setUp(self):
        self.manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    def test_cuda_sources_are_shipped(self):
        """Without them the sdist cannot build the fused kernels, and the
        documented `FORCE_CUDA=1 pip install --no-build-isolation .` path dies
        on a missing mgn/csrc/bindings.cpp."""
        self.assertRegex(self.manifest, r"recursive-include mgn/csrc .*\.cu")
        for src in self._setup_sources():
            self.assertTrue((ROOT / src).is_file(),
                            f"setup.py lists a source that does not exist: {src}")

    def test_compiled_artifacts_are_never_included(self):
        for line in self.manifest.splitlines():
            line = line.strip()
            if line.startswith(("include ", "recursive-include ", "graft ")):
                self.assertNotRegex(line, r"\.(pyd|so|dll|dylib)\b",
                                    "MANIFEST.in asks for a compiled artifact")
        self.assertRegex(self.manifest, r"global-exclude .*\.pyd")
        self.assertRegex(self.manifest, r"global-exclude .*\.pth")

    def test_research_scaffolding_is_pruned(self):
        for rel in ("tests", "data", "paper_and_documents", "mgn/tests"):
            self.assertIn(f"prune {rel}", self.manifest)

    @staticmethod
    def _setup_sources():
        text = (ROOT / "setup.py").read_text(encoding="utf-8")
        return re.findall(r'"(mgn/csrc/[^"]+\.(?:cu|cpp|h))"', text)


class TestVersionSingleSourcing(unittest.TestCase):

    def test_dunder_versions_match_pyproject(self):
        """`__version__` is hand-written in two packages next to the real one
        in pyproject.toml; the docs and the PyPI page quote the latter."""
        version = PROJECT["version"]
        for rel in ("mgn/accel/__init__.py", "hts_lib/__init__.py"):
            text = (ROOT / rel).read_text(encoding="utf-8")
            match = re.search(r'^__version__ = "([^"]+)"', text, re.MULTILINE)
            self.assertIsNotNone(match, f"{rel} lost its __version__")
            self.assertEqual(match.group(1), version,
                             f"{rel} disagrees with pyproject.toml")


class TestReleaseAutomation(unittest.TestCase):

    def setUp(self):
        self.workflow = WORKFLOW.read_text(encoding="utf-8")
        self.guide = (ROOT / "RELEASING.md").read_text(encoding="utf-8")

    def test_uploads_use_trusted_publishing_not_a_token(self):
        self.assertIn("id-token: write", self.workflow)
        self.assertIn("pypa/gh-action-pypi-publish", self.workflow)
        self.assertNotRegex(self.workflow, r"(?i)password\s*[:=]")
        self.assertNotRegex(self.workflow, r"(?i)\b(api[_-]?token|twine_password|pypi_token)\b")

    def test_environments_match_the_guide(self):
        """PyPI matches the publisher configuration on owner / repo / workflow
        / environment. A rename on one side only is a release-time failure."""
        for env in ("pypi", "testpypi"):
            self.assertIn(f"name: {env}", self.workflow)
            self.assertIn(env, self.guide)
        self.assertIn("publish.yml", self.guide)

    def test_publishing_is_gated_on_verification(self):
        self.assertIn("needs: verify", self.workflow)
        self.assertIn("twine check", self.workflow)
        self.assertIn("py3-none-any.whl", self.workflow,
                      "the pure-Python wheel guard is gone")

    def test_ci_exercises_the_console_script(self):
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("aml-selfcheck", ci)


class TestInstallDocs(unittest.TestCase):

    def test_console_script_points_at_a_real_main(self):
        script = PROJECT["scripts"]["aml-selfcheck"]
        module, _, attr = script.partition(":")
        self.assertEqual(attr, "main")
        rel = pathlib.Path(*module.split(".")).with_suffix(".py")
        self.assertTrue((ROOT / rel).is_file(), f"{script} has no {rel}")
        self.assertIn("def main(", (ROOT / rel).read_text(encoding="utf-8"))

    def test_readme_documents_the_published_name(self):
        """The README is the PyPI description; if it never names the
        distribution, nobody can find the install command after landing there."""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"pip install {PROJECT['name']}", readme)
        self.assertIn("aml-selfcheck", readme)

    def test_readme_links_the_release_guide(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("RELEASING.md", readme)


if __name__ == "__main__":
    unittest.main()
