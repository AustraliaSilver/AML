import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .architecture import GeneratedModel
from .constraints import ResourceBudget, phi_cost
from .regularization import similarity_regularization, arch_distance, contrastive_architecture_loss


class MetaLoss(nn.Module):
    def __init__(
        self,
        lambda_p: float = 1e-4,
        lambda_c: float = 1e-6,
        lambda_m: float = 1e-7,
        lambda_t: float = 0.0,
        lambda_e: float = 0.0,
        lambda_balance: float = 0.0,
        lambda_contrastive: float = 0.0,
        lambda_stability: float = 0.0,
        lambda_design: float = 0.0,
        lambda_reg: float = 0.0,
    lambda_eff: float = 0.0,
    # Default = SmallCNN's parameter count from experiments/results_fashion_mnist.json,
    # i.e. the efficiency barrier is calibrated against the strongest static baseline
    # by default. Override per-experiment via MetaLoss(..., p_allowance=...).
    p_allowance: float = 110986.0,
        reg_sigma: float = 1.0,
        resource_budget: ResourceBudget = None,
    ):
        super().__init__()
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.lambda_m = lambda_m
        self.lambda_t = lambda_t
        self.lambda_e = lambda_e
        self.lambda_balance = lambda_balance
        self.lambda_contrastive = lambda_contrastive
        self.lambda_stability = lambda_stability
        self.lambda_design = lambda_design
        self.lambda_reg = lambda_reg
        self.lambda_eff = lambda_eff
        self.p_allowance = p_allowance
        self.reg_sigma = reg_sigma
        self.resource_budget = resource_budget

    def forward(
        self,
        logits_train: torch.Tensor,
        targets_train: torch.Tensor,
        logits_val: torch.Tensor,
        targets_val: torch.Tensor,
        model: GeneratedModel,
        prev_model: GeneratedModel = None,
        pi_logits: torch.Tensor = None,
        task_ids: torch.Tensor = None,
        models_batch: list = None,
        task_loss_fn: nn.Module = None,
        **kwargs
    ) -> dict:
        if task_loss_fn is None:
            task_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

        L_task = task_loss_fn(logits_train, targets_train)
        
        if logits_val is not None and targets_val is not None:
            L_val = task_loss_fn(logits_val, targets_val)
            
            # ============================================================
            # G_gap: Normalized Generalization Gap (Hybrid Stable Version)
            # Numerator: clamp(Delta, 0) - only penalizes overfitting, stable gradient
            # Denominator: softplus(ln(Y) - L_train) - C-infinity, avoids division by 0
            # ============================================================
            # Derive ln(#classes) from the structure instead of hard-coding
            # ln(10): the gap normalization is silently wrong on any
            # dataset with a different label count.
            ln_Y = math.log(max(int(model.structure.output_dim), 2))
            gap_numerator   = torch.clamp(L_val - L_task.detach(), min=0.0)
            gap_denominator = F.softplus(ln_Y - L_task.detach())
            G_gap = gap_numerator / (gap_denominator + 1e-4)
        else:
            G_gap = torch.tensor(0.0, device=logits_train.device)

        L_resource = phi_cost(
            model.structure,
            self.lambda_p,
            self.lambda_c,
            self.lambda_m,
            self.resource_budget,
            lambda_t=self.lambda_t,
            lambda_e=self.lambda_e,
        ).mean()

        L_eff = torch.tensor(0.0, device=logits_train.device)
        O_val = torch.tensor(0.0, device=logits_train.device)
        if getattr(self, "lambda_eff", 0.0) > 0:
            from .constraints import compute_aml_efficiency_torch
            probs = F.softmax(logits_train, dim=-1)
            true_probs = probs.gather(1, targets_train.to(logits_train.device).unsqueeze(1)).squeeze(1).mean()
            P_eff = model.structure.effective_params_formula()
            
            E_aml_raw = compute_aml_efficiency_torch(true_probs, getattr(self, "p_allowance", 110986.0), P_eff).mean()
            E_aml_raw = torch.clamp(E_aml_raw, min=1e-6)
            
            # ============================================================
            # Normalized E_AML: 1 - exp(-k * E_aml) in (0, 1)
            # Ensures O is always in (0,1) -> (1-O) always positive -> loss
            # can never become an unbounded reward mechanism.
            # k = 5.0: scaling factor controlling sensitivity
            # ============================================================
            # ============================================================
            k = 5.0
            E_aml = 1.0 - torch.exp(-k * E_aml_raw)
            
            # Curriculum Compression:
            epoch_progress = kwargs.get("epoch_progress", 0.0)
            if epoch_progress <= 0.33:
                E_0 = 0.4
            elif epoch_progress <= 0.66:
                decay_ratio = (epoch_progress - 0.33) / 0.33
                E_0 = 0.4 - decay_ratio * (0.4 - 0.1)
            else:
                E_0 = 0.1
                
            # Unified AML Objective (Final Form)
            # O = E_hat * exp(-lambda * G_gap / (E_hat/E0)^gamma)
            
            # Anneal lambda_eff based on epoch_progress
            progress = kwargs.get("epoch_progress", 0.0)
            if progress < 0.2:
                current_lambda_eff = 0.0
            elif progress < 0.5:
                current_lambda_eff = self.lambda_eff * (progress - 0.2) / 0.3
            else:
                current_lambda_eff = self.lambda_eff
                
            gamma = 2.0
            lambda_val = 1.0  # Reduced penalty to avoid gradient vanishing from batch noise
            eps = 1e-4

            R_gamma = (E_aml / E_0) ** gamma
            # Use first-order G_gap instead of second-order to avoid noise amplification
            O = E_aml * torch.exp(-lambda_val * G_gap / (R_gamma + eps))
            O_val = O

            # Loss = L_task + mu * (1 - O), with O in (0,1) so (1-O) >= 0
            L_eff = current_lambda_eff * (1.0 - O)

            # ============================================================
            # Direct Over-Budget Penalty (gradient-aware)
            # F_norm saturates to 0 when over budget, killing gradient.
            # This quadratic penalty provides constant gradient signal
            # to push P_eff below P_allow.
            # ============================================================
            r_raw = P_eff.mean() / getattr(self, "p_allowance", 110986.0)
            over_budget = torch.relu(r_raw - 1.0)
            L_eff = L_eff + current_lambda_eff * over_budget * over_budget

        L_design = torch.tensor(0.0, device=logits_train.device)
        if self.lambda_design > 0:
            L_design = self.lambda_design * (
                model.structure.effective_params_formula().mean() / 100
            )

        L_stability = torch.tensor(0.0, device=logits_train.device)
        if self.lambda_stability > 0 and prev_model is not None:
            L_stability = self.lambda_stability * arch_distance(model, prev_model)

        L_balance = torch.tensor(0.0, device=logits_train.device)
        if self.lambda_balance > 0 and pi_logits is not None:
            pi = F.softmax(pi_logits, dim=-1)
            pi_avg = pi.mean(dim=0)
            K = pi_avg.shape[0]
            entropy = -(pi_avg * torch.log(pi_avg + 1e-8)).sum()
            L_balance = -self.lambda_balance * entropy

        L_contrastive = torch.tensor(0.0, device=logits_train.device)
        if self.lambda_contrastive > 0 and models_batch is not None and task_ids is not None:
            L_contrastive = self.lambda_contrastive * contrastive_architecture_loss(
                models_batch, task_ids
            )

        loss = L_task + L_resource + L_eff + L_design + L_stability + L_balance + L_contrastive

        return {
            "loss": loss,
            "task_loss": L_task,
            "resource_cost": L_resource.detach(),
            "aml_objective": O_val,
            "efficiency_cost": L_eff.detach(),
            "design_cost": L_design.detach(),
            "stability_cost": L_stability.detach(),
            "balance_cost": L_balance.detach(),
            "contrastive_cost": L_contrastive.detach(),
        }

    def compute_similarity_reg(self, models: list, embeddings: torch.Tensor) -> torch.Tensor:
        if self.lambda_reg <= 0 or len(models) < 2:
            return torch.tensor(0.0, device=embeddings.device)
        return similarity_regularization(models, embeddings, self.reg_sigma, self.lambda_reg)
