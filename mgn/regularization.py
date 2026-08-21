import torch
import torch.nn.functional as F

from .architecture import ModelStructure, GeneratedModel


def kernel_rbf(emb_i: torch.Tensor, emb_j: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    dist_sq = torch.cdist(emb_i, emb_j, p=2).pow(2)
    return torch.exp(-dist_sq / (2 * sigma * sigma))


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
    if g1.dim() == 2 and g2.dim() == 2 and g1.shape[0] == 1 and g2.shape[0] == 1:
        gate_dist = F.mse_loss(g1, g2, reduction="sum")

    return L_diff + D_diff + gate_dist


def weight_distance(w1: dict, w2: dict) -> torch.Tensor:
    dist = torch.tensor(0.0, device=next(iter(w1.values())).device)
    for key in w1:
        if key in w2:
            t1, t2 = w1[key], w2[key]
            if t1.shape == t2.shape:
                dist = dist + F.mse_loss(t1, t2, reduction="mean")
            else:
                min_dims = [min(s1, s2) for s1, s2 in zip(t1.shape, t2.shape)]
                s1 = t1[tuple(slice(d) for d in min_dims)]
                s2 = t2[tuple(slice(d) for d in min_dims)]
                dist = dist + F.mse_loss(s1, s2, reduction="mean")
    return dist


def arch_distance(a1: GeneratedModel, a2: GeneratedModel) -> torch.Tensor:
    # As weights are dynamically generated per-sample, comparing them across different batches is meaningless.
    # Therefore, we only compare structural gating decisions.
    return structure_distance(a1.structure, a2.structure)


def similarity_regularization(
    models: list,
    embeddings: torch.Tensor,
    sigma: float = 1.0,
    lambda_reg: float = 1.0,
) -> torch.Tensor:
    n = len(models)
    if n < 2:
        return torch.tensor(0.0, device=embeddings.device)
    K = kernel_rbf(embeddings, embeddings, sigma)
    reg = torch.tensor(0.0, device=embeddings.device)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = structure_distance(models[i].structure, models[j].structure)
            reg = reg + K[i, j] * d
            count += 1
    return lambda_reg * reg / max(count, 1)


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


def arch_entropy(arch: GeneratedModel, eps: float = 1e-8) -> torch.Tensor:
    total_entropy = torch.tensor(0.0, device=next(iter(arch.weights.values())).device)
    for w in arch.weights.values():
        w_flat = w.flatten()
        if w_flat.numel() < 2:
            continue
        std = w_flat.std() + eps
        normalized = w_flat / std
        probs = F.softmax(normalized, dim=-1)
        entropy = -(probs * torch.log(probs + eps)).sum()
        total_entropy = total_entropy + entropy
    return total_entropy
