import time
import torch
import torch.nn as nn
from ..models.losses import compute_layer_aware_diversity_loss


class HtSBasisIsolator:
    """
    Continual Learning Parameter Isolator for HtS Private Residual Basis Experts.
    Enables No-Replay Continual Learning:
    - Task i gets exclusive access to its allocated expert basis slots [s, e)
    - Past experts [0, s) are frozen (gradient masked + snapshot preserved)
    - Future experts [e, K) are frozen to avoid router dependency contamination
    """
    def __init__(self, model):
        self.model = model
        self.basis_sys = model.basis_sys
        self.K = model.K
        self.active_range = (0, self.K)
        self.snapshots = {}
        self.masks = {}

        # Register hook on all private basis tensors
        for name in ("b1_priv", "b1_2_priv", "b2_priv", "b2_3_priv", "b3_priv"):
            param = getattr(self.basis_sys, name, None)
            if param is not None:
                mask = torch.ones_like(param)
                self.masks[name] = mask
                self._attach_hook(param, mask)

    def _attach_hook(self, param, mask):
        def hook(grad):
            return grad * mask
        param.register_hook(hook)

    def begin_task(self, task_idx, num_tasks=3):
        """
        Calculates expert range [s, e) for the current task and updates gradient masks.
        """
        s = round(task_idx * self.K / num_tasks)
        e = round((task_idx + 1) * self.K / num_tasks)
        self.active_range = (s, e)

        # Snapshot current basis parameters
        for name in self.masks:
            param = getattr(self.basis_sys, name)
            self.snapshots[name] = param.data.clone()
            # Mask out non-active experts
            mask = self.masks[name]
            mask.fill_(0.0)
            if e > s:
                mask[s:e].fill_(1.0)
        return s, e

    @torch.no_grad()
    def restore(self):
        """
        Safely restores frozen experts to their exact post-task snapshots.
        """
        for name, snap in self.snapshots.items():
            param = getattr(self.basis_sys, name)
            mask = self.masks[name]
            param.data.copy_(torch.where(mask > 0, param.data, snap))


class ContinualTrainer:
    """
    Continual Learning Trainer with State-Preserving Optimizer,
    Streaming Co-evolutionary Replay, No-Replay Basis Isolation,
    and Portable CPU/CUDA Autocast.
    """
    def __init__(self, model, device=None, lr=1.5e-3, weight_decay=3e-4, 
                 lambda_ent=0.015, lambda_use=0.10, lambda_ort=0.05,
                 preserve_optimizer_state=True, enable_basis_isolation=True):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
        
        self.use_amp = (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_ent = lambda_ent
        self.lambda_use = lambda_use
        self.lambda_ort = lambda_ort
        self.preserve_optimizer_state = preserve_optimizer_state
        self.enable_basis_isolation = enable_basis_isolation
        self.isolator = HtSBasisIsolator(self.model) if enable_basis_isolation else None
        
        # Persistent optimizer across task sequences
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train_task(self, current_task, train_loader, epochs=10, replay_loaders=None, replay_weights=None, lr=None, task_idx=None, num_tasks=3, use_tc=False):
        """
        Trains on current_task with optional replay from past task loaders and basis isolation.
        """
        replay_loaders = replay_loaders or {}
        replay_weights = replay_weights or {}
        task_lr = lr or self.lr
        
        if self.isolator is not None and task_idx is not None:
            s, e = self.isolator.begin_task(task_idx, num_tasks=num_tasks)
            print(f"[HtS Isolator] Task {task_idx} active basis experts: [{s}, {e}) of K={self.model.K}")

        if not self.preserve_optimizer_state:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=task_lr, weight_decay=self.weight_decay)
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = task_lr
                
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        replay_iters = {task: iter(loader) for task, loader in replay_loaders.items()}
        
        print(f"\n[START TRAINING] Task: {current_task.upper()} ({epochs} Epochs) | Interleaved Replay: {list(replay_loaders.keys())}")
        
        for ep in range(1, epochs + 1):
            self.model.train()
            t0 = time.time()
            tau = max(0.5, 1.0 - (ep / epochs) * 0.5)
            
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    # 1. Primary task loss
                    logits, alphas = self.model(x, task=current_task, tau=tau, use_tc=use_tc)
                    loss_main = self.criterion(logits, y)
                    l_ent, l_use, l_ort = compute_layer_aware_diversity_loss(alphas, self.model, K=self.model.K)
                    
                    total_loss = loss_main + self.lambda_ent * l_ent + self.lambda_use * l_use + self.lambda_ort * l_ort
                    
                    # 2. Interleaved Replay losses
                    for r_task, r_weight in replay_weights.items():
                        try:
                            x_r, y_r = next(replay_iters[r_task])
                        except StopIteration:
                            replay_iters[r_task] = iter(replay_loaders[r_task])
                            x_r, y_r = next(replay_iters[r_task])
                        x_r, y_r = x_r.to(self.device, non_blocking=True), y_r.to(self.device, non_blocking=True)
                        
                        logits_r, _ = self.model(x_r, task=r_task, tau=tau, use_tc=use_tc)
                        loss_r = self.criterion(logits_r, y_r)
                        total_loss += r_weight * loss_r

                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    self.optimizer.step()

                if self.isolator is not None and task_idx is not None:
                    self.isolator.restore()
                
            sch.step()
            print(f"  * Epoch {ep:2d}/{epochs:2d} ({time.time()-t0:4.1f}s) completed.")
            
        print(f"[COMPLETED] Task {current_task.upper()}\n")
