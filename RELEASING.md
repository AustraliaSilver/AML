# Releasing `aml-framework` to PyPI

Publication is automated by [`.github/workflows/publish.yml`](.github/workflows/publish.yml).
This file is the human half of that automation: the one-time account setup, then
the steps for each release.

---

## 1. What gets published

| Artifact | Contents | Who installs it |
| --- | --- | --- |
| `aml_framework-<v>-py3-none-any.whl` | Pure-Python `mgn` + `hts_lib` | Everyone: `pip install aml-framework` |
| `aml_framework-<v>.tar.gz` (sdist) | The same, plus `mgn/csrc/*.cu`, `build_ext.py` and the quickstarts | Anyone who wants the fused CUDA kernels |

The optional fused kernels (`mgn.accel._C`, `_Cin`) are **never** shipped in the
wheel. They are compiled against a specific CUDA toolkit, MSVC/GCC and Python
ABI, so a single wheel cannot serve both Linux and Windows, and a Windows `.pyd`
inside a tag reading `py3-none-any` breaks every other platform. `setup.py`
therefore builds nothing under `pip`, prints a one-line notice saying so, and
`mgn.accel` falls back to the numerically equivalent PyTorch GroupedConv path.
Section 5 explains how to build them from source.

Nothing else leaves the repository: not the datasets, checkpoints, the audit
log, `experiments/`, `probes/` or `paper_and_documents/`.

---

## 2. One-time setup

### 2.1 Accounts

1. Create an account on [pypi.org](https://pypi.org/account/register/) and
   enable 2FA (mandatory for uploading).
2. Create the same on [test.pypi.org](https://test.pypi.org/account/register/)
   — it is a separate database with separate credentials.

### 2.2 Trusted Publishers (no API token anywhere)

Both the workflow and the index must agree on five strings. On PyPI go to
**Account settings → Publishing → Add a new pending publisher** and enter:

| Field | Value |
| --- | --- |
| PyPI Project Name | `aml-framework` |
| Owner | `AustraliaSilver` |
| Repository name | `AML` |
| Workflow name | `publish.yml` |
| Environment name | `pypi` |

Repeat on **test.pypi.org** with the environment name `testpypi`. (With
TestPyPI and `aml-framework`, the project name may be taken by someone else;
if so, only TestPyPI needs adjusting — see the note at the end of 2.3.)

If the repository or the GitHub account is ever renamed, update these five
fields in the same commit as the rename, or every release fails on the OIDC
exchange.

### 2.3 GitHub environments

In the repository: **Settings → Environments → New environment**, create `pypi`
and `testpypi`. Optionally add a required reviewer to `pypi` so a release waits
for a human click. The names must match 2.2 exactly.

> If the `aml-framework` project name is unavailable on TestPyPI, publish the
> dry run to PyPI with `skip-existing` instead, or temporarily point the
> `publish-testpypi` job at a different `repository-url`. Only the name matters;
> the wheel itself is unaffected.

---

## 3. Cutting a release

1. Bump `version` in `pyproject.toml`, and the two `__version__` literals that
   mirror it (`mgn/accel/__init__.py`, `hts_lib/__init__.py`). A test fails if
   they disagree with `pyproject.toml`.
2. Commit and push to `main`; wait for **CI** to go green.
3. Tag the commit with a `v` prefix and push the tag:

   ```bash
   git tag v2.0.1
   git push origin v2.0.1
   ```

4. The **Publish** workflow runs automatically: `verify` (full test suite +
   `aml-selfcheck`) → `build` (sdist + wheel, `twine check`, pure-Python guard)
   → `publish-pypi`. Watch it with:

   ```bash
   gh run list --workflow=Publish
   gh run watch
   ```

   The tag is checked against the packaged version, so a mislabelled release
   fails before it reaches the index.

### Dry run first

To exercise the whole path without touching PyPI: **Actions → Publish → Run
workflow → target: `testpypi`**. Then install the result:

```bash
pip install -i https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ aml-framework
aml-selfcheck
```

The extra index is required because TestPyPI does not host the `torch`
dependency.

> A published version can never be re-uploaded, even after deleting it. If
> `2.0.1` is broken, *yank* it on PyPI and ship `2.0.2` — never try to reuse
> the number.

---

## 4. Installing the release

```bash
pip install aml-framework      # pulls the CPU build of torch
aml-selfcheck                  # imports, forwards and backprops; exit code 0 = healthy
```

A CUDA build of torch is a plain pip concern, not an AML one — install it
first and `pip` will not touch it afterwards:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install aml-framework
python -c "import torch; print(torch.cuda.is_available())"
```

`aml-selfcheck` reports whether the fused kernels are present and says they are
optional when they are not.

---

## 5. Building the fused CUDA kernels

From a source checkout or the sdist, on a host with `nvcc` and a C++ compiler:

```bash
# inside the environment that already has torch (no build isolation: torch is a
# runtime, not a build, requirement, so the isolated env has no torch headers)
FORCE_CUDA=1 pip install --no-build-isolation .
```

`FORCE_CUDA=1` (or `AML_BUILD_CUDA=1`) also builds on a machine with no visible
GPU, such as a Docker build stage; without it `torch.cuda.is_available()` is
`False` and the extensions are skipped, which then fails at import time rather
than build time. Alternatively, from an editable checkout:

```bash
python build_ext.py            # both kernels
python build_ext.py _C         # just the GroupNorm variant
python -m pip install --no-build-isolation -e ".[dev]"
```

On Windows, run from a Developer Command Prompt (or after `vcvars64.bat`) so
`cl.exe` is on `PATH`; `build_ext.py` also finds MSVC itself when it can.

---

## 6. Building the packages locally

```bash
python -m pip install --upgrade build twine
pyproject-build            # console script -> dist/
python -m twine check --strict dist/*
```

Two local gotchas:

- `python -m build` **from the repository root fails** with *"'build' is a
  package and cannot be directly executed"*, because the git-ignored `./build/`
  directory shadows the installed `build` module on `sys.path[0]`. Use the
  `pyproject-build` console script, or delete `./build` first. CI never sees
  this: a fresh checkout has no `./build`.
- An editable install leaves `aml_framework.egg-info/SOURCES.txt` in the working
  tree, and setuptools reads that stale file list as the starting point for the
  next sdist. `MANIFEST.in` prunes the directories it can name (the `tests/`
  tree, `data/`, the paper, the local `build/`); anything else that file
  remembers will ship. `git clean -Xdf` before building is the safe habit. The
  `egg-info` directory itself is always regenerated and shipped inside the
  sdist — that is standard, not staleness.
