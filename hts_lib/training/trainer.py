import time
import torch
import torch.nn as nn
from ..models.losses import compute_layer_aware_diversity_loss

class ContinualTrainer:
    """
    Continual Learning Trainer orchestrator for HtS Hypernetwork.
    Executes sequential task training with Co-evolutionary Streaming Replay and Regularization.
    """
    def __init__(self, model, device=None, lr=1.5e-3, weight_decay=3e-4, 
                 lambda_ent=0.015, lambda_use=0.10, lambda_ort=0.05):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.03)
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.lambda_ent = lambda_ent
        self.lambda_use = lambda_use
        self.lambda_ort = lambda_ort

    def train_task(self, current_task, train_loader, epochs=10, replay_loaders=None, replay_weights=None):
        """
        Trains on current_task while optionally interleaving replay data from past tasks.
        """
        replay_loaders = replay_loaders or {}
        replay_weights = replay_weights or {}
        
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
        
        replay_iters = {task: iter(loader) for task, loader in replay_loaders.items()}
        
        print(f"\n[START TRAINING] Task: {current_task.upper()} ({epochs} Epochs) | Replay Tasks: {list(replay_loaders.keys())}")
        
        for ep in range(1, epochs + 1):
            self.model.train()
            t0 = time.time()
            tau = max(0.5, 1.0 - (ep / epochs) * 0.5)
            
            for x, y in train_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                opt.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    # 1. Main task loss
                    logits, alphas = self.model(x, task=current_task, tau=tau)
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
                        
                        logits_r, _ = self.model(x_r, task=r_task, tau=tau)
                        loss_r = self.criterion(logits_r, y_r)
                        total_loss += r_weight * loss_r

                self.scaler.scale(total_loss).backward()
                self.scaler.step(opt)
                self.scaler.update()
                
            sch.step()
            print(f"  * Epoch {ep:2d}/{epochs:2d} ({time.time()-t0:4.1f}s) completed.")
            
        print(f"[COMPLETED] Task {current_task.upper()}\n")
