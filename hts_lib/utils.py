import os
import random
import numpy as np
import torch

def set_seed(seed=2026, deterministic=True):
    """
    Sets deterministic seeds across Python, NumPy, and PyTorch (CPU/CUDA).
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(device_override=None):
    """
    Returns available torch.device (CUDA or CPU).
    """
    if device_override is not None:
        return torch.device(device_override)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
