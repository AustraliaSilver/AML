import torch
import torch.nn as nn
import torch.nn.functional as F
from .basis import HtSDualResidualBasis
from .blocks import HtSResidualBlock

class HtSContinualModel(nn.Module):
    """
    HtS Continual Learning Hypernetwork Architecture.
    
    Features:
    - Multi-domain native stems (MNIST, CIFAR-10, Fashion-MNIST, or custom).
    - Shared Base + Dynamic Residual Basis with K=8 Experts.
    - 8 Multi-Stage Residual Blocks (D = 24 -> 48 -> 96).
    - Task-specific classification heads.
    """
    def __init__(self, d1=24, d2=48, d3=96, K=8, num_classes=10, supported_tasks=None):
        super().__init__()
        self.K = K
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.supported_tasks = supported_tasks or ["mnist", "cifar", "fashion"]
        
        # 1. Native Stems
        self.stems = nn.ModuleDict({
            "mnist": nn.Sequential(
                nn.Conv2d(1, d1, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(4, d1),
                nn.GELU()
            ),
            "cifar": nn.Sequential(
                nn.Conv2d(3, d1, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(4, d1),
                nn.GELU()
            ),
            "fashion": nn.Sequential(
                nn.Conv2d(1, d1, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(4, d1),
                nn.GELU()
            )
        })
        
        # 2. Dual-Residual Basis System
        self.basis_sys = HtSDualResidualBasis(K=K, d1=d1, d2=d2, d3=d3)
        
        # 3. Task Controller & Router
        self.task_ctrl = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(d1 * 4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 8 * K)
        )
        
        # 4. Multi-Stage Residual Blocks
        self.b1 = HtSResidualBlock(d1, d1, stride=1)
        self.b2 = HtSResidualBlock(d1, d1, stride=1)
        self.b3 = HtSResidualBlock(d1, d2, stride=2)
        self.b4 = HtSResidualBlock(d2, d2, stride=1)
        self.b5 = HtSResidualBlock(d2, d2, stride=1)
        self.b6 = HtSResidualBlock(d2, d3, stride=2)
        self.b7 = HtSResidualBlock(d3, d3, stride=1)
        self.b8 = HtSResidualBlock(d3, d3, stride=1)
        
        # 5. Task-specific Classification Heads
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Dropout(0.1),
                nn.Linear(d3, num_classes)
            ) for task in self.supported_tasks
        })

    def add_task(self, task_name, in_channels=3, num_classes=10):
        """
        Dynamically registers a new domain/task with its own native stem and classification head.
        """
        if task_name not in self.stems:
            self.stems[task_name] = nn.Sequential(
                nn.Conv2d(in_channels, self.d1, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(4, self.d1),
                nn.GELU()
            )
            self.heads[task_name] = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Dropout(0.1),
                nn.Linear(self.d3, num_classes)
            )
            self.supported_tasks.append(task_name)

    def forward(self, x, task="cifar", tau=0.8):
        """
        Forward execution with on-the-fly weight synthesis.
        """
        B = x.shape[0]
        if task not in self.stems:
            raise KeyError(f"Task '{task}' not registered. Available: {list(self.stems.keys())}")
            
        h = self.stems[task](x)
        logits_alpha = self.task_ctrl(h).view(B, 8, self.K)
        
        if self.training:
            alphas = F.gumbel_softmax(logits_alpha, tau=tau, hard=False, dim=-1)
        else:
            alphas = F.softmax(logits_alpha / tau, dim=-1)
            
        # Stage 1 (D=24)
        h = self.b1(h, self.basis_sys.get_kernel('b1', alphas[:, 0]))
        h = self.b2(h, self.basis_sys.get_kernel('b1', alphas[:, 1]))
        
        # Stage 2 (D=48)
        h = self.b3(h, self.basis_sys.get_kernel('b1_2', alphas[:, 2]))
        h = self.b4(h, self.basis_sys.get_kernel('b2', alphas[:, 3]))
        h = self.b5(h, self.basis_sys.get_kernel('b2', alphas[:, 4]))
        
        # Stage 3 (D=96)
        h = self.b6(h, self.basis_sys.get_kernel('b2_3', alphas[:, 5]))
        h = self.b7(h, self.basis_sys.get_kernel('b3', alphas[:, 6]))
        h = self.b8(h, self.basis_sys.get_kernel('b3', alphas[:, 7]))
        
        logits = self.heads[task](h)
        return logits, alphas

    def count_parameters_breakdown(self, task="cifar"):
        """
        Returns breakdown of active child parameters vs total hypernetwork parameters.
        """
        child_active = (sum(p.numel() for p in self.stems[task].parameters()) +
                        sum(p.numel() for l in [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8] for p in l.parameters()) +
                        sum(p.numel() for p in self.heads[task].parameters()))
        
        basis_params = sum(p.numel() for p in self.basis_sys.parameters())
        ctrl_params = sum(p.numel() for p in self.task_ctrl.parameters())
        total = sum(p.numel() for p in self.parameters())
        
        return {
            "child_active_params": child_active,
            "basis_params": basis_params,
            "router_params": ctrl_params,
            "total_hypernetwork_params": total
        }
