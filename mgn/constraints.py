from typing import Optional
import torch

from .architecture import ModelStructure


class ResourceBudget:
    def __init__(self, P_max=None, C_max=None, M_max=None):
        self.P_max = P_max
        self.C_max = C_max
        self.M_max = M_max


def compute_aml_efficiency(accuracy: float, p_allowance: float, q_used: float) -> float:
    """
    Compute E_AML = (a / log10(p^2 / q)) * (p / q)^alpha(a, b)
    where alpha(a, b) is the C^infinity polynomial + tanh formula ensuring smooth bounded derivatives across a in [0, 1].
    """
    import numpy as np
    import math
    if accuracy <= 0 or q_used <= 0 or p_allowance <= 0:
        return 0.0
    a = min(max(float(accuracy), 1e-3), 1.0)
    # Mirror the torch twin's guards: q >= 100 and r = q/p clamped to 4.5
    # (beta*(r-1) stays below the float64-safe exp range). Without these,
    # math.exp(20*(r-1)) raises OverflowError for heavily over-budget q.
    q = min(max(float(q_used), 100.0), float(p_allowance) * 4.5)
    p = float(p_allowance)
    b = max(p / q, 0.1)

    e = math.e
    inv_e = 1.0 / e
    term1 = (0.5 / (e - 1.0)) * ((e / 2.0) * (a ** 2) - a)
    diff = a - inv_e
    term2 = 2.0 * (diff ** 2) * (1.0 + 1.0 * math.tanh(5.0 * diff)) * (1.0 - 1.0 / b)
    alpha_bal = term1 + term2 + 0.05

    p2_q = (p * p) / q
    denom = np.log10(p2_q) if p2_q > 1 else 1e-3
    ratio_term = min(max(b ** alpha_bal, 0.01), 100.0)
    
    # Apply C^inf Normalized Exponential Soft Barrier
    beta = 20.0
    lam = 2.0
    r = q / p
    F_norm = ( (1.0 + math.exp(beta * (r - 1.0))) / 2.0 ) ** (-lam / beta)
    
    return float((a / denom) * ratio_term * F_norm)


def compute_aml_efficiency_torch(accuracy: torch.Tensor, p_allowance: float, q_used: torch.Tensor) -> torch.Tensor:
    """
    Differentiable tensor computation of E_new(a, p, q) using the C^infinity polynomial + tanh formula
    and the normalized Softplus barrier F_norm.
    """
    import math
    a = torch.clamp(accuracy.float(), min=1e-3, max=1.0)
    q = torch.clamp(q_used.float(), min=100.0)
    p = float(p_allowance) if p_allowance > 0 else 1e5
    b = torch.clamp(p / q, min=0.1)

    e = math.e
    inv_e = 1.0 / e
    term1 = (0.5 / (e - 1.0)) * ((e / 2.0) * (a ** 2) - a)
    diff = a - inv_e
    term2 = 2.0 * (diff ** 2) * (1.0 + 1.0 * torch.tanh(5.0 * diff)) * (1.0 - 1.0 / b)
    alpha_bal = term1 + term2 + 0.05

    p2_q = (p * p) / q
    # clamp(log10, 1e-3) == where(p2_q>1, log10, 1e-3) clamped: for
    # p2_q<=1 log10<=0 so the clamp floor produces the identical 1e-3.
    # The old form built a fresh CUDA tensor scalar PER CALL -- every H2D
    # scalar constructor serializes the stream behind queued work.
    denom = torch.clamp(torch.log10(p2_q), min=1e-3)
    ratio_term = torch.clamp(b ** alpha_bal, min=0.01, max=100.0)
    
    # Apply C^inf Normalized Exponential Soft Barrier
    beta = 20.0
    lam = 2.0
    r = q / p
    r_clamped = torch.clamp(r, max=4.5)  # beta * (4.5 - 1.0) = 70.0 (safe from float32 exp overflow)
    F_norm = ( (1.0 + torch.exp(beta * (r_clamped - 1.0))) / 2.0 ) ** (-lam / beta)
    
    return (a / denom) * ratio_term * F_norm


def phi_cost(
    structure: ModelStructure,
    lambda_p: float = 1e-4,
    lambda_c: float = 1e-6,
    lambda_m: float = 1e-7,
    budget: Optional[ResourceBudget] = None,
    lambda_t: float = 0.0,
    lambda_e: float = 0.0,
) -> torch.Tensor:
    P = structure.effective_params_formula()
    C = structure.flops_per_sample()
    M = structure.memory_bytes()
    T = C / 1e6
    E = C * 1e-8
    cost = lambda_p * P + lambda_c * C + lambda_m * M + lambda_t * T + lambda_e * E
    if budget is not None:
        if budget.P_max is not None:
            cost = cost + torch.relu(P - budget.P_max)
        if budget.C_max is not None:
            cost = cost + torch.relu(C - budget.C_max)
        if budget.M_max is not None:
            cost = cost + torch.relu(M - budget.M_max)
    return cost
