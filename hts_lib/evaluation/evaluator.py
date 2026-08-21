import time
import torch
import torch.nn.functional as F

class ContinualEvaluator:
    """
    Evaluator suite for multi-domain continual learning.
    Measures Accuracy, Latency, Pairwise Cosine Diversity, and Router Distributions.
    """
    def __init__(self, model, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def evaluate_task(self, loader, task="cifar"):
        self.model.eval()
        correct, total = 0, 0
        t0 = time.time()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.amp.autocast('cuda'):
                    logits, _ = self.model(x, task=task, tau=0.5)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        elapsed = time.time() - t0
        acc = 100.0 * correct / total
        errors = total - correct
        return {
            "accuracy": acc,
            "errors": errors,
            "total": total,
            "correct": correct,
            "latency_seconds": elapsed,
            "fps": total / elapsed if elapsed > 0 else 0
        }

    def measure_cosine_diversity(self, task_loaders):
        """
        Computes intra-task and inter-task cosine similarity of synthesized weights.
        """
        self.model.eval()
        weights_dict = {}
        alphas_dict = {}
        
        with torch.no_grad():
            for task, loader in task_loaders.items():
                x, _ = next(iter(loader))
                x = x.to(self.device)
                _, alphas = self.model(x, task=task, tau=0.5)
                
                # Weight tensor of deepest block (Stage 3 Block 8)
                w = self.model.basis_sys.get_kernel('b3', alphas[:, 7]).flatten(1)
                w_norm = F.normalize(w, p=2, dim=-1)
                
                weights_dict[task] = w_norm
                alphas_dict[task] = alphas[:, 7].mean(dim=0).cpu().numpy().round(3).tolist()
                
        metrics = {"router_alphas": alphas_dict}
        tasks = list(task_loaders.keys())
        
        # Intra-task cosine
        for t in tasks:
            w = weights_dict[t]
            cos_mat = torch.mm(w, w.t())
            mask = ~torch.eye(w.size(0), dtype=torch.bool, device=self.device)
            metrics[f"intra_cosine_{t}"] = float(cos_mat[mask].mean().item())
            
        # Inter-task cosine
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                t1, t2 = tasks[i], tasks[j]
                cos_inter = torch.mm(weights_dict[t1], weights_dict[t2].t()).mean().item()
                metrics[f"inter_cosine_{t1}_vs_{t2}"] = float(cos_inter)
                
        return metrics
