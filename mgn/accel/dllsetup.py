"""Portable native-library bootstrap for the fused CUDA extensions.

On Windows, Python 3.8+ no longer searches PATH for DLL dependencies of
extension modules, so the CUDA runtime and torch's own DLLs must be
registered explicitly via os.add_dll_directory before importing _C/_Cin.

This module derives those directories from the *live* environment
(the installed torch package and the CUDA_PATH / CUDA_HOME env vars)
instead of hardcoding absolute developer-machine paths.

It also installs a defensive patch around torch._inductor's write_atomic,
which can raise FileExistsError when several processes populate the
inductor cache concurrently.
"""

from __future__ import annotations

import glob
import os
import sys

__all__ = ["setup_dll_directories", "patch_inductor_cache", "bootstrap"]

_dll_done = False
_inductor_done = False


def _candidate_cuda_bins():
    """Yield plausible CUDA bin directories, most specific first."""
    for var in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        root = os.environ.get(var)
        if root:
            yield os.path.join(root, "bin")
    # Fall back to whatever toolkits are installed, newest version first.
    pattern = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"
    for root in sorted(glob.glob(pattern), reverse=True):
        yield os.path.join(root, "bin")


def _torch_lib_dir():
    try:
        import torch
    except ImportError:
        return None
    return os.path.join(os.path.dirname(torch.__file__), "lib")


def setup_dll_directories():
    """Register CUDA + torch DLL directories. No-op off Windows."""
    global _dll_done
    if _dll_done or sys.platform != "win32":
        _dll_done = True
        return []

    added = []
    for d in list(_candidate_cuda_bins()) + [_torch_lib_dir()]:
        if d and os.path.isdir(d):
            try:
                os.add_dll_directory(d)
                added.append(d)
            except (OSError, AttributeError):
                pass
        if added and d == _torch_lib_dir():
            break
    _dll_done = True
    return added


def patch_inductor_cache():
    """Make torch._inductor's cache writes tolerant of concurrent writers."""
    global _inductor_done
    if _inductor_done:
        return
    _inductor_done = True
    try:
        import pathlib

        import torch._inductor.codecache as _cc
    except Exception:
        return

    _orig = getattr(_cc, "write_atomic", None)
    if _orig is None or getattr(_orig, "_aml_patched", False):
        return

    def _patched(path, data, **kwargs):
        try:
            _orig(path, data, **kwargs)
        except FileExistsError:
            pathlib.Path(path).unlink(missing_ok=True)
            _orig(path, data, **kwargs)

    _patched._aml_patched = True
    _cc.write_atomic = _patched


def bootstrap(suppress_dynamo_errors=True):
    """Full pre-import setup used by the training entry points."""
    setup_dll_directories()
    patch_inductor_cache()
    if suppress_dynamo_errors:
        try:
            import torch._dynamo

            torch._dynamo.config.suppress_errors = True
        except Exception:
            pass
