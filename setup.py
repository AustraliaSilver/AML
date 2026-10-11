"""
setup.py - Standard packaging for AML with CUDA extensions.

Enables:
    pip install -e .
    python setup.py build_ext --inplace
"""

import os
import sys
from setuptools import setup, find_packages

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Safely import torch & C++ extension builder if available
cmdclass = {}
ext_modules = []

try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    has_torch = True
except ImportError:
    has_torch = False

if has_torch and torch.cuda.is_available():
    cmdclass = {"build_ext": BuildExtension}
    extra_compile_args = {
        "cxx": ["/EHsc", "/std:c++17", "/Zc:preprocessor"] if sys.platform == "win32" else ["-std=c++17"],
        "nvcc": [
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ],
    }

    if sys.platform == "win32":
        extra_compile_args["nvcc"].extend([
            "-Xcompiler", "/EHsc",
            "-Xcompiler", "/wd4068",
            "-Xcompiler", "/wd4067",
            "-Xcompiler", "/wd4624",
            "-Xcompiler", "/wd4190",
            "-Xcompiler", "/wd4018",
            "-Xcompiler", "/wd4275",
            "-Xcompiler", "/wd4267",
            "-Xcompiler", "/wd4244",
            "-Xcompiler", "/wd4251",
            "-Xcompiler", "/wd4819",
            "-Xcompiler", "/MD",
            "-Xcompiler", "/Zc:preprocessor",
        ])

    csrc_utils = os.path.join(ROOT_DIR, "mgn", "csrc", "utils")

    ext_modules = [
        CUDAExtension(
            name="mgn.accel._C",
            sources=[
                "mgn/csrc/bindings.cpp",
                "mgn/csrc/kernels/custom_dynamic_ops.cu",
                "mgn/csrc/kernels/fused_pipeline.cu",
                "mgn/csrc/kernels/fused_backward.cu",
                "mgn/csrc/kernels/fused_chain_conv.cu",
                "mgn/csrc/kernels/batched_hyper_linear.cu",
                "mgn/csrc/kernels/fused_post_conv.cu",
                "mgn/csrc/kernels/fused_norm_relu_gate.cu",
            ],
            include_dirs=[csrc_utils],
            extra_compile_args=extra_compile_args,
        ),
        CUDAExtension(
            name="mgn.accel._Cin",
            sources=[
                "mgn/csrc/in/bindings_in.cpp",
                "mgn/csrc/kernels/custom_dynamic_ops.cu",
                "mgn/csrc/in/fused_pipeline_in.cu",
                "mgn/csrc/in/fused_backward_in.cu",
            ],
            include_dirs=[csrc_utils],
            extra_compile_args=extra_compile_args,
        ),
    ]

setup(
    name="aml",
    version="2.0.0",
    author="AML Team",
    description="Adaptive Model Learning with Native CUDA Acceleration & Continual Learning",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.9",
)
