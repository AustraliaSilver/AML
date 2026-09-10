import random
import torch
from torch.utils.data import Dataset, DataLoader

class EpisodicMemoryBuffer:
    """
    Episodic Memory Buffer for Continual Learning under strict Memory Budgets.

    Balanced per-class exemplar storage is used when the source dataset
    exposes a `.targets` attribute (torchvision datasets do); otherwise the
    subset is plain uniform random. There is no reservoir/stream API --
    exemplars are drawn from the full in-memory training dataset.

    Args:
        max_total_samples: Maximum number of total samples across all past tasks (e.g. 500, 1000, 2000).
        per_task_budget: Optional fixed budget per task. If None, budget is equally partitioned across tasks.
    """
    def __init__(self, max_total_samples=1000, per_task_budget=None):
        self.max_total_samples = max_total_samples
        self.per_task_budget = per_task_budget
        self.task_memories = {}  # {task_name: {"data": List[Tensor], "targets": List[Tensor]}}

    def update_with_dataset(self, task_name, dataset, num_samples=None):
        """
        Populates exemplar memory for `task_name` from a full training dataset.
        Class-balanced when `dataset.targets` exists, random otherwise.
        """
        budget = num_samples or self.per_task_budget or (self.max_total_samples // (len(self.task_memories) + 1))

        targets = getattr(dataset, "targets", None)
        if targets is not None:
            # torch RNG keeps this reproducible under set_seed()
            t = torch.as_tensor(list(targets))
            quota = max(budget // int(t.max().item()) + 1, 1)
            selected_idx = []
            for c in torch.unique(t):
                cls_idx = (t == c).nonzero(as_tuple=True)[0]
                keep = cls_idx[torch.randperm(len(cls_idx))[:quota]]
                selected_idx.extend(keep.tolist())
            selected_idx = selected_idx[:budget]
        else:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            selected_idx = indices[:budget]
        
        exemplar_data = []
        exemplar_targets = []
        for idx in selected_idx:
            x, y = dataset[idx]
            exemplar_data.append(x)
            exemplar_targets.append(y)
            
        self.task_memories[task_name] = {
            "data": torch.stack(exemplar_data),
            "targets": torch.tensor(exemplar_targets, dtype=torch.long)
        }
        
        # Re-balance past tasks if budget is dynamic
        if self.per_task_budget is None:
            self._rebalance_memory()

    def _rebalance_memory(self):
        num_tasks = len(self.task_memories)
        if num_tasks == 0:
            return
        budget_per_task = self.max_total_samples // num_tasks
        for task_name in self.task_memories:
            mem = self.task_memories[task_name]
            if len(mem["data"]) > budget_per_task:
                mem["data"] = mem["data"][:budget_per_task]
                mem["targets"] = mem["targets"][:budget_per_task]

    def get_memory_loader(self, task_name, batch_size=64):
        """
        Returns a DataLoader over the episodic memory buffer for `task_name`.
        """
        if task_name not in self.task_memories:
            return None
        mem = self.task_memories[task_name]
        dataset = torch.utils.data.TensorDataset(mem["data"], mem["targets"])
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    def get_total_stored_samples(self):
        return sum(len(mem["data"]) for mem in self.task_memories.values())

    def get_summary(self):
        return {
            task: len(mem["data"]) for task, mem in self.task_memories.items()
        }
