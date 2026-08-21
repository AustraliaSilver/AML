import torch
import torch.nn as nn

from .architecture import GeneratedModel
from .constraints import phi_cost


class ModelEvaluator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 64,
        lambda_p: float = 1e-4,
        lambda_c: float = 1e-6,
        lambda_m: float = 1e-7,
        lambda_t: float = 1e-8,
        lambda_e: float = 1e-10,
        quality_weight: float = 1.0,
        cost_weight: float = 1.0,
    ):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.lambda_m = lambda_m
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e
        self.quality_weight = quality_weight
        self.cost_weight = cost_weight

        self.quality_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def score_candidate(self, model: GeneratedModel) -> torch.Tensor:
        z = model.embedding
        quality = self.quality_head(z).squeeze(-1).mean()

        cost = phi_cost(
            model.structure,
            lambda_p=self.lambda_p,
            lambda_c=self.lambda_c,
            lambda_m=self.lambda_m,
        ).mean()

        T = model.structure.flops_per_sample().mean() / 1e6
        E = model.structure.flops_per_sample().mean() * 1e-8
        cost = cost + self.lambda_t * T + self.lambda_e * E

        return self.quality_weight * quality - self.cost_weight * cost

    def compute_efficiency(self, accuracy: float, p_allowance: float, model: GeneratedModel) -> float:
        """
        Evaluate candidate architecture using E_new(a, p, q):
        E(a, p, q) = (a / log10(p^2 / q)) * (p / q)^|a * log10(a)|
        """
        from .constraints import compute_aml_efficiency
        q_used = model.structure.effective_params_formula().mean().item()
        return compute_aml_efficiency(accuracy, p_allowance, q_used)

    def select_best(
        self,
        candidates: list,
    ) -> tuple:
        scores = torch.stack([self.score_candidate(m) for m in candidates])
        best_idx = scores.argmax().item()
        return candidates[best_idx], best_idx, scores
