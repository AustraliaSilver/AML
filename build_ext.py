"""build_ext.py -- build the fused CUDA extensions from source.

Builds and deploys both native extensions used by the AML executor:

  _C    (mgn/csrc/bindings.cpp + mgn/csrc/kernels/*.cu)
        GroupNorm-1 fused pipeline, grouped conv fwd/bwd, maxpool, post-conv.

  _Cin  (mgn/csrc/in/*)
        InstanceNorm variant of the fused pipeline (per-channel norm stats).

Usage:
    python build_ext.py            # build both
    python build_ext.py _C         # build only _C
    python build_ext.py _Cin       # build only _Cin

The compiled artifacts land in mgn/accel/ as
`<name>.cp<py><py>-<platform>.pyd` (Windows) or `.so` (Linux) so that
`import mgn.accel._C` resolves natively without a JIT recompile.

Compiled extensions are intentionally NOT tracked in git (see .gitignore) --
they are build artifacts. This script is the reproducible way to regenerate
them, so a fresh clone is never missing an irrecoverable binary.
"""

import argparse
import os
import shutil
import sys
import sysconfig

ROOT = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(ROOT, "mgn", "csrc")
ACCEL = os.path.join(ROOT, "mgn", "accel")

EXTENSIONS = {
    "_C": [
        os.path.join(CSRC, "bindings.cpp"),
        os.path.join(CSRC, "kernels", "custom_dynamic_ops.cu"),
        os.path.join(CSRC, "kernels", "fused_pipeline.cu"),
        os.path.join(CSRC, "kernels", "fused_backward.cu"),
        os.path.join(CSRC, "kernels", "batched_hyper_linear.cu"),
    ],
    "_Cin": [
        os.path.join(CSRC, "in", "bindings_in.cpp"),
        os.path.join(CSRC, "kernels", "custom_dynamic_ops.cu"),
        os.path.join(CSRC, "in", "fused_pipeline_in.cu"),
        os.path.join(CSRC, "in", "fused_backward_in.cu"),
    ],
}


def ext_suffix():
    """Return the platform-correct extension suffix, e.g. .cp312-win_amd64.pyd"""
    return sysconfig.get_config_var("EXT_SUFFIX") or ".pyd"


def compiler_flags():
    """Host/CUDA compiler flags, adjusted per platform.

    On Windows, recent MSVC + CUDA toolkits require the standard-conforming
    preprocessor (`/Zc:preprocessor`); without it nvcc aborts with
    "MSVC/cl.exe with traditional preprocessor is used".
    """
    cflags = []
    cuda_flags = ["-O3", "--use_fast_math"]
    if sys.platform == "win32":
        cflags += ["/Zc:preprocessor"]
        cuda_flags += ["-Xcompiler", "/Zc:preprocessor"]
    else:
        cflags += ["-O3"]
    return cflags, cuda_flags


def _setup_windows_msvc():
    import shutil
    import subprocess
    import glob
    if shutil.which("cl") is not None:
        return
    candidates = (
        glob.glob(r"C:\Program Files*\Microsoft Visual Studio\*\BuildTools\VC\Auxiliary\Build\vcvars64.bat") +
        glob.glob(r"C:\Program Files*\Microsoft Visual Studio\*\Community\VC\Auxiliary\Build\vcvars64.bat") +
        glob.glob(r"C:\Program Files*\Microsoft Visual Studio\*\Enterprise\VC\Auxiliary\Build\vcvars64.bat") +
        glob.glob(r"C:\Program Files*\Microsoft Visual Studio\*\Professional\VC\Auxiliary\Build\vcvars64.bat")
    )
    if not candidates:
        return
    vcvars = candidates[0]
    try:
        cmd = f'cmd.exe /s /c "\"{vcvars}\" >nul && set"'
        out = subprocess.check_output(cmd, shell=True, text=True)
        for line in out.splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                os.environ[k] = v
        print(f"[build_ext] loaded MSVC environment from {vcvars}")
    except Exception as e:
        print(f"[build_ext] warning: failed to load MSVC environment: {e}")


def build(name, verbose=True):
    if sys.platform == "win32":
        _setup_windows_msvc()
    import torch.utils.cpp_extension as ce
    ce._check_cuda_version = lambda compiler_name, compiler_version: None
    from torch.utils.cpp_extension import load

    sources = EXTENSIONS[name]
    missing = [s for s in sources if not os.path.isfile(s)]
    if missing:
        raise FileNotFoundError(
            f"{name}: missing source file(s):\n  " + "\n  ".join(missing))

    cflags, cuda_flags = compiler_flags()
    print(f"\n=== building {name} ({len(sources)} sources) ===")
    mod = load(
        name=name,
        sources=sources,
        extra_include_paths=[os.path.join(CSRC, "utils")],
        extra_cflags=cflags,
        extra_cuda_cflags=cuda_flags,
        verbose=verbose,
    )
    print(f"built: {mod.__file__}")

    dst = os.path.join(ACCEL, name + ext_suffix())
    shutil.copy2(mod.__file__, dst)
    print(f"deployed: {dst}")
    return dst


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("targets", nargs="*", default=None,
                    choices=list(EXTENSIONS) + [],
                    help="extensions to build (default: all)")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="suppress nvcc/compiler output")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        sys.exit("torch is required to build the extensions: pip install torch")

    if not torch.cuda.is_available():
        print("[warn] CUDA runtime not visible to torch. nvcc is still required "
              "to compile; the built module will only be importable on a CUDA host.")

    targets = args.targets or list(EXTENSIONS)
    for name in targets:
        build(name, verbose=not args.quiet)

    print("\n=== verifying native imports ===")
    for name in targets:
        mod = __import__(f"mgn.accel.{name}", fromlist=[name])
        exports = [x for x in dir(mod) if not x.startswith("_")]
        print(f"  {name}: OK ({len(exports)} exports) -> {mod.__file__}")


if __name__ == "__main__":
    main()
