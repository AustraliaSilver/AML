"""
HtS Library: Dynamic Hypernetwork with Dual-Residual Basis & Continual Learning
================================================================================
A state-of-the-art PyTorch library for dynamic weight synthesis, multi-domain continual learning,
and hypernetwork regularization.
"""

from .models.basis import HtSDualResidualBasis
from .models.blocks import HtSResidualBlock
from .models.hypernetwork import HtSContinualModel
from .models.losses import compute_layer_aware_diversity_loss
from .data.loaders import get_continual_dataloaders
from .training.trainer import ContinualTrainer
from .evaluation.evaluator import ContinualEvaluator

__version__ = "1.0.0"
__all__ = [
    "HtSDualResidualBasis",
    "HtSResidualBlock",
    "HtSContinualModel",
    "compute_layer_aware_diversity_loss",
    "get_continual_dataloaders",
    "ContinualTrainer",
    "ContinualEvaluator"
]
