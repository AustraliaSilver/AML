"""aml_accel: CUDA-accelerated kernels for Adaptive Model Learning."""

from .functions import (
    fused_dynamic_conv2d,
)

__version__ = "2.0.0"

__all__ = [
    "fused_dynamic_conv2d",
]
