import time
import numpy as np
import torch
import torch.nn.functional as F

class ContinualEvaluator:
    """
    Formal Academic Evaluator Suite for Continual Learning.
    
    Computes:
    - Task Accuracy Matrix A[i, j] (Accuracy on Task j after completing Training Stage i)
    - Average Final Accuracy (ACC)
    - Backward Transfer (BWT)
    - Average Forgetting (F)
    - Forward Transfer (FWT)
    - Dynamic Kernel Diversity metrics (Cosine Similarity)
    """
    def __init__(self, model, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.use_amp = (self.device.type == 'cuda')

    def evaluate_task(self, loader, task="cifar"):
        self.model.eval()
        correct, total = 0, 0
        t0 = time.time()
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    logits, _ = self.model(x, task=task, tau=0.5)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        elapsed = time.time() - t0
        acc = 100.0 * correct / total if total > 0 else 0.0
        errors = total - correct
        return {
            "accuracy": acc,
            "errors": errors,
            "total": total,
            "correct": correct,
            "latency_seconds": elapsed,
            "fps": total / elapsed if elapsed > 0 else 0
        }

    def evaluate_all_tasks(self, test_loaders):
        """
        Evaluates current model state across all registered tasks.
        Returns: dict {task_name: accuracy}
        """
        results = {}
        for task_name, loader in test_loaders.items():
            res = self.evaluate_task(loader, task=task_name)
            results[task_name] = res["accuracy"]
        return results

    @staticmethod
    def compute_continual_metrics(acc_matrix, task_names):
        """
        Calculates formal continual learning metrics from Accuracy Matrix A.
        
        Args:
            acc_matrix: 2D numpy array or list of lists of shape [T, T]
                        A[i, j] = Accuracy on Task j after completing Task i.
            task_names: List of task names of length T.
            
        Returns:
            dict containing ACC, BWT, Forgetting, and structured summary.
        """
        A = np.array(acc_matrix, dtype=float)
        T = len(task_names)
        
        # 1. Final Average Accuracy: (1/T) * sum_{j=1}^T A[T-1, j]
        final_accs = A[T - 1, :]
        avg_acc = float(np.mean(final_accs))
        
        # 2. Backward Transfer (BWT): (1/(T-1)) * sum_{i=1}^{T-1} (A[T-1, i] - A[i, i])
        if T > 1:
            bwt_terms = [A[T - 1, i] - A[i, i] for i in range(T - 1)]
            bwt = float(np.mean(bwt_terms))
            
            # 3. Forgetting: (1/(T-1)) * sum_{i=1}^{T-1} max_{l in {1...T-1}} (A[l, i] - A[T-1, i])
            forgetting_terms = [
                max(0.0, np.max(A[:T - 1, i]) - A[T - 1, i])
                for i in range(T - 1)
            ]
            avg_forgetting = float(np.mean(forgetting_terms))
        else:
            bwt = 0.0
            avg_forgetting = 0.0
            
        return {
            "task_names": task_names,
            "accuracy_matrix": A.tolist(),
            "final_task_accuracies": {task_names[i]: round(final_accs[i], 2) for i in range(T)},
            "average_accuracy": round(avg_acc, 2),
            "backward_transfer_bwt": round(bwt, 2),
            "average_forgetting": round(avg_forgetting, 2)
        }

    def measure_dynamic_kernel_diversity(self, task_loaders):
        """
        Computes pairwise cosine similarity of synthesized weights (Diagnostic Metric).
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
        
        for t in tasks:
            w = weights_dict[t]
            cos_mat = torch.mm(w, w.t())
            mask = ~torch.eye(w.size(0), dtype=torch.bool, device=self.device)
            metrics[f"intra_cosine_{t}"] = float(cos_mat[mask].mean().item())
            
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                t1, t2 = tasks[i], tasks[j]
                cos_inter = torch.mm(weights_dict[t1], weights_dict[t2].t()).mean().item()
                metrics[f"inter_cosine_{t1}_vs_{t2}"] = float(cos_inter)
                
        return metrics
