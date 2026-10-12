import torch
import torch.nn as nn
import torch.nn.functional as F
from .basis import HtSDualResidualBasis
from .blocks import HtSResidualBlock

# Orthogonality/entropy stage weights, indexed by stage (1..3). Kept identical
# to the historical compute_layer_aware_diversity_loss per-block weighting.
STAGE_WEIGHTS = (0.1, 0.5, 1.0)

def _default_layout(stage_blocks):
    """Expand (n_stage1, n_stage2, n_stage3) into per-block
    (kernel, in_d, out_d, stride) specs against the fixed 3-stage dual basis.
    stage_blocks=(2,2,2) reproduces the original 8-block network exactly."""
    n1, n2, n3 = stage_blocks
    layout = [("b1", "d1", "d1", 1)] * n1
    layout.append(("b1_2", "d1", "d2", 2))
    layout += [("b2", "d2", "d2", 1)] * n2
    layout.append(("b2_3", "d2", "d3", 2))
    layout += [("b3", "d3", "d3", 1)] * n3
    # Per-block stage weight: intra-stage blocks use their stage weight;
    # a transition block carries the weight of the stage it enters.
    weights = ([STAGE_WEIGHTS[0]] * n1 + [STAGE_WEIGHTS[1]]
               + [STAGE_WEIGHTS[1]] * n2 + [STAGE_WEIGHTS[2]]
               + [STAGE_WEIGHTS[2]] * n3)
    return layout, weights


class HtSContinualModel(nn.Module):
    """
    HtS Continual Learning Hypernetwork Architecture.

    Features:
    - Multi-domain native stems (MNIST, CIFAR-10, Fashion-MNIST, or custom).
    - Shared Base + Dynamic Residual Basis with K experts.
    - Three-stage depth, parameterized via stage_blocks (residual blocks per
      stage); depth is no longer fixed to 8. stage_blocks=(2,2,2) is the
      historical 8-block network.
    - Task-specific classification heads.
    """
    def __init__(self, d1=24, d2=48, d3=96, K=8, num_classes=10, supported_tasks=None,
                 stage_blocks=(2, 2, 2)):
        super().__init__()
        self.K = K
        self.d1 = d1
        self.d2 = d2
        self.d3 = d3
        self.stage_blocks = tuple(stage_blocks)
        self.supported_tasks = supported_tasks or ["mnist", "cifar", "fashion"]

        dims = {"d1": d1, "d2": d2, "d3": d3}
        layout, weights = _default_layout(self.stage_blocks)
        self.n_blocks = len(layout)
        self.block_kernels = [k for (k, _i, _o, _s) in layout]
        # Non-persistent buffer: moves with .to(device) but stays out of
        # state_dict, so checkpoints remain compatible with the historical
        # b1..b8 model layout from which this metadata is derived.
        self.register_buffer("block_weights", torch.tensor(weights, dtype=torch.float32), persistent=False)

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

        # 3. Task Controller & Router (one K-way alpha per block)
        self.task_ctrl = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(d1 * 4, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, self.n_blocks * K)
        )

        # 4. Multi-Stage Residual Blocks (depth from stage_blocks)
        self.blocks = nn.ModuleList([
            HtSResidualBlock(dims[i], dims[o], stride=s) for (_k, i, o, s) in layout
        ])

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
            # Determine existing device and dtype
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype

            stem = nn.Sequential(
                nn.Conv2d(in_channels, self.d1, 3, stride=2, padding=1, bias=False),
                nn.GroupNorm(4, self.d1),
                nn.GELU()
            ).to(device=device, dtype=dtype)

            head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Dropout(0.1),
                nn.Linear(self.d3, num_classes)
            ).to(device=device, dtype=dtype)

            self.stems[task_name] = stem
            self.heads[task_name] = head
            self.supported_tasks.append(task_name)

    def forward(self, x, task="cifar", tau=0.8, use_tc=False):
        """
        Forward execution with on-the-fly weight synthesis.
        """
        B = x.shape[0]
        if task not in self.stems:
            raise KeyError(f"Task '{task}' not registered. Available: {list(self.stems.keys())}")

        stem_conv = self.stems[task][0]
        if hasattr(stem_conv, "in_channels") and x.shape[1] != stem_conv.in_channels:
            if x.shape[1] == 3 and stem_conv.in_channels == 1:
                x = x[:, :1, :, :]
            elif x.shape[1] == 1 and stem_conv.in_channels == 3:
                x = x.repeat(1, 3, 1, 1)

        h = self.stems[task](x)
        logits_alpha = self.task_ctrl(h).view(B, self.n_blocks, self.K)

        if self.training:
            alphas = F.gumbel_softmax(logits_alpha, tau=tau, hard=False, dim=-1)
        else:
            alphas = F.softmax(logits_alpha / tau, dim=-1)

        for idx, blk in enumerate(self.blocks):
            h = blk(h, self.basis_sys.get_kernel(self.block_kernels[idx], alphas[:, idx], use_tc=use_tc))

        logits = self.heads[task](h)
        return logits, alphas

    def count_parameters_breakdown(self, task="cifar"):
        """
        Returns breakdown of active child parameters vs total hypernetwork parameters.
        """
        child_active = (sum(p.numel() for p in self.stems[task].parameters()) +
                        sum(p.numel() for blk in self.blocks for p in blk.parameters()) +
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
