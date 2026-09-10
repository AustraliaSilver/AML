import torch
import torch.nn.functional as F

from .architecture import ModelStructure, GeneratedModel


def structure_distance(s1: ModelStructure, s2: ModelStructure) -> torch.Tensor:
    L1 = s1.num_layers
    L2 = s2.num_layers
    D1 = s1.hidden_dim
    D2 = s2.hidden_dim

    if L1.dim() > 0 or L2.dim() > 0:
        L1 = L1.mean()
        L2 = L2.mean()
    if D1.dim() > 0 or D2.dim() > 0:
        D1 = D1.mean()
        D2 = D2.mean()

    L_diff = (L1 - L2).pow(2)
    D_diff = 0.01 * (D1 - D2).pow(2)

    gate_dist = 0.0
    g1 = s1.layer_gates
    g2 = s2.layer_gates
    if g1.dim() == 2 and g2.dim() == 2:
        # The old `shape[0] == 1 for both` condition meant the per-layer gate
        # term never fired for batched training (the usual case). Compare the
        # overlapping rows instead.
        n = min(g1.shape[0], g2.shape[0])
        gate_dist = F.mse_loss(g1[:n], g2[:n], reduction="mean")

    return L_diff + D_diff + gate_dist


def arch_distance(a1: GeneratedModel, a2: GeneratedModel) -> torch.Tensor:
    # As weights are dynamically generated per-sample, comparing them across different batches is meaningless.
    # Therefore, we only compare structural gating decisions.
    return structure_distance(a1.structure, a2.structure)


def contrastive_architecture_loss(
    models: list,
    task_ids: torch.Tensor,
) -> torch.Tensor:
    n = len(models)
    if n < 2:
        return torch.tensor(0.0, device=models[0].weights["fc1.weight"].device)
    device = models[0].weights["fc1.weight"].device
    sum_same = torch.tensor(0.0, device=device)
    sum_diff = torch.tensor(0.0, device=device)
    count_same = 0
    count_diff = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = arch_distance(models[i], models[j])
            if (task_ids[i] == task_ids[j]).item():
                sum_same = sum_same + d
                count_same += 1
            else:
                sum_diff = sum_diff + d
                count_diff += 1
    mean_same = sum_same / max(count_same, 1)
    mean_diff = sum_diff / max(count_diff, 1)
    return mean_same / (mean_diff + 1e-8)
