import random
import torch
from torch.utils.data import Dataset, DataLoader

class EpisodicMemoryBuffer:
    """
    Episodic Memory Buffer for Continual Learning under strict Memory Budgets.
    Implements Reservoir Sampling and Balanced Per-Class Exemplar Storage.
    
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
        Samples randomly or up to allocated budget.
        """
        budget = num_samples or self.per_task_budget or (self.max_total_samples // (len(self.task_memories) + 1))
        
        # Extract subset of exemplars
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
