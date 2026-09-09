import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .generator import ModelGenerator
from .executor import TaskNetwork
from .losses import MetaLoss
from .evaluator import ModelEvaluator
from .architecture import ModelStructure, GeneratedModel
from .regularization import arch_distance


class MetaTrainer:
    def __init__(
        self,
        generator: ModelGenerator,
        executor: TaskNetwork,
        meta_loss: MetaLoss,
        evaluator: ModelEvaluator = None,
        lr: float = 1e-3,
        device: str = "cpu",
        num_candidates: int = 1,
        optimizer: str = "adamw",
        weight_decay: float = 0.05,
    ):
        self.generator = generator.to(device)
        self.executor = executor.to(device)
        self.meta_loss = meta_loss
        self.evaluator = evaluator.to(device) if evaluator is not None else None
        self.device = device
        self.num_candidates = num_candidates

        params = list(self.generator.parameters())
        if self.evaluator is not None:
            params = params + list(self.evaluator.parameters())
        opt_cls = {"adamw": torch.optim.AdamW,
                   "adam": torch.optim.Adam}.get(optimizer.lower())
        if opt_cls is None:
            raise ValueError(f"Unknown optimizer {optimizer!r}; "
                             "expected 'adamw' or 'adam'.")
        self.optimizer = opt_cls(params, lr=lr, weight_decay=weight_decay)

        self.history = {"loss": [], "task_loss": [], "resource_cost": [], "acc": []}
        self._prev_model = None

    def train_epoch(self, dataloader: DataLoader, val_dataloader: DataLoader = None,
                    tau: float = 1.0, epoch_progress: float = 0.0,
                    task_loss_fn: nn.Module = None) -> dict:
        self.generator.train()
        self.executor.train()
        total_metrics = {"loss": 0.0, "task_loss": 0.0, "resource_cost": 0.0, "acc": 0.0}
        num_batches = 0
        val_iter = iter(val_dataloader) if val_dataloader is not None else None

        for batch in dataloader:
            if len(batch) == 3:
                x_batch, y_batch, task_ids = batch
            else:
                x_batch, y_batch = batch
                task_ids = None
            x_batch = x_batch.to(self.device)
            if self.generator.training and x_batch.dim() == 2 and x_batch.shape[1] == 3072:
                # Vectorized per-sample hflip from the TORCH RNG so runs stay
                # reproducible under torch.manual_seed (python's `random` was
                # left unseeded by most experiment entry points).
                x_images = x_batch.view(-1, 3, 32, 32)
                flip = torch.rand(x_images.size(0), device=x_images.device) < 0.5
                x_images = torch.where(flip.view(-1, 1, 1, 1), x_images.flip(-1), x_images)
                x_batch = x_images.view(-1, 3072)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            if val_iter is not None:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_dataloader)
                    val_batch = next(val_iter)
                
                if len(val_batch) == 3:
                    val_x, val_y, _ = val_batch
                else:
                    val_x, val_y = val_batch
                
                # Data augmentation for val_batch to match format
                if val_x.dim() == 4 and val_x.shape[2] == 32:
                    val_x = val_x.view(-1, 3072)
                
                val_x = val_x.to(self.device)
                val_y = val_y.to(self.device)
                
                x_train = x_batch
                y_train = y_batch
                x_val = val_x
                y_val = val_y
                
                # Ensure batch sizes match
                min_size = min(x_train.shape[0], x_val.shape[0])
                x_train = x_train[:min_size]
                y_train = y_train[:min_size]
                x_val = x_val[:min_size]
                y_val = y_val[:min_size]
            else:
                x_train, x_val = x_batch.chunk(2)
                y_train, y_val = y_batch.chunk(2)
            
            if self.num_candidates > 1 and self.evaluator is not None:
                candidates = [self.generator(x_train, tau=tau) for _ in range(self.num_candidates)]
                model, best_idx, scores = self.evaluator.select_best(candidates)
            else:
                model = self.generator(x_train, tau=tau)

            logits_train = self.executor(x_train, model)
            logits_val = self.executor(x_val, model)

            kwargs = {"pi_logits": model.pi_logits, "epoch_progress": epoch_progress}
            if task_loss_fn is not None:
                kwargs["task_loss_fn"] = task_loss_fn
            if self.meta_loss.lambda_stability > 0 and self._prev_model is not None:
                kwargs["prev_model"] = self._prev_model
            if self.meta_loss.lambda_contrastive > 0 and task_ids is not None:
                n_contrast = min(16, x_train.shape[0])
                models_batch = [self.generator(x_train[i:i+1], tau=tau) for i in range(n_contrast)]
                kwargs["models_batch"] = models_batch
                kwargs["task_ids"] = task_ids[:n_contrast].to(self.device)

            metrics = self.meta_loss(logits_train, y_train, logits_val, y_val, model, **kwargs)
            total_loss = metrics["loss"]

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
            self.optimizer.step()

            with torch.no_grad():
                self._prev_model = GeneratedModel(
                    structure=ModelStructure(
                        layer_gates=model.structure.layer_gates.clone(),
                        dim_weights=model.structure.dim_weights.clone(),
                        residual=model.structure.residual.clone(),
                        dim_masks=model.structure.dim_masks.clone() if model.structure.dim_masks is not None else None,
                        block_mode=model.structure.block_mode.clone() if model.structure.block_mode is not None else None,
                        input_dim=model.structure.input_dim,
                        output_dim=model.structure.output_dim,
                        max_dim=model.structure.max_dim,
                        max_layers=model.structure.max_layers,
                        dim_choices=model.structure.dim_choices,
                    ),
                    weights={k: v.clone() for k, v in model.weights.items()},
                )

            preds = logits_train.argmax(dim=-1)
            acc = (preds == y_train).float().mean()

            total_metrics["loss"] += total_loss.item()
            total_metrics["task_loss"] += metrics["task_loss"].item()
            total_metrics["resource_cost"] += metrics["resource_cost"].item()
            total_metrics["acc"] += acc.item()
            total_metrics.setdefault("aml_objective", 0.0)
            total_metrics["aml_objective"] += metrics.get("aml_objective", 0.0).item()
            num_batches += 1

        if num_batches == 0:
            return {k: 0.0 for k in dict(total_metrics, aml_objective=0.0)}
        avg = {k: v / num_batches for k, v in total_metrics.items()}
        for k, v in avg.items():
            self.history.setdefault(k, []).append(v)
        return avg

    def train(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        val_x: torch.Tensor = None,
        val_y: torch.Tensor = None,
        epochs: int = 100,
        batch_size: int = 32,
        tau_schedule: bool = True,
    ):
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if val_x is not None and val_y is not None:
            val_dataset = TensorDataset(val_x, val_y)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        self._prev_model = None

        import sys
        print_every = 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        for epoch in range(epochs):
            epoch_progress = epoch / max(1, epochs - 1)
            tau = max(0.5, 1.0 - epoch / 30.0) if tau_schedule else 1.0
            metrics = self.train_epoch(loader, val_dataloader=val_loader, tau=tau, epoch_progress=epoch_progress)
            scheduler.step()
            if (epoch + 1) % print_every == 0 or epoch == epochs - 1:
                print(
                    f"  [ConvAML] Epoch {epoch+1}/{epochs} | "
                    f"loss={metrics['loss']:.4f} | "
                    f"task={metrics['task_loss']:.4f} | "
                    f"acc={metrics['acc']:.4f} | "
                    f"O={metrics['aml_objective']:.4f}"
                )
                sys.stdout.flush()

    def predict_with_evaluator(self, x: torch.Tensor, num_candidates: int = 5,
                               tau: float = 1.0) -> tuple:
        if self.evaluator is None:
            raise ValueError("predict_with_evaluator requires an evaluator; "
                             "pass one to MetaTrainer.__init__ or use predict().")
        gen_was = self.generator.training
        exe_was = self.executor.training
        self.generator.eval()
        self.executor.eval()
        if self.evaluator is not None:
            self.evaluator.eval()
        try:
            with torch.no_grad():
                candidates = [self.generator(x.to(self.device), tau=tau)
                              for _ in range(num_candidates)]
                best, best_idx, scores = self.evaluator.select_best(candidates)
                logits = self.executor(x.to(self.device), best)
                return logits.argmax(dim=-1), best_idx, scores
        finally:
            self.generator.train(gen_was)
            self.executor.train(exe_was)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        gen_was = self.generator.training
        exe_was = self.executor.training
        self.generator.eval()
        self.executor.eval()
        try:
            with torch.no_grad():
                model = self.generator(x.to(self.device))
                logits = self.executor(x.to(self.device), model)
                return logits.argmax(dim=-1)
        finally:
            self.generator.train(gen_was)
            self.executor.train(exe_was)

    def generate_architecture(self, x: torch.Tensor) -> dict:
        gen_was = self.generator.training
        self.generator.eval()
        try:
            with torch.no_grad():
                model = self.generator(x.to(self.device))
                S = model.structure
                L_mean = S.num_layers.mean().item()
                D_mean = S.hidden_dim.mean().item()
                P_eff = S.effective_params_formula().mean().item()
                return {
                    "L": round(L_mean, 1),
                    "D": round(D_mean, 1),
                    "P_eff": int(round(P_eff)),
                }
        finally:
            # Without this, one eval call between epochs silently leaves the
            # generator in eval mode and the next train_epoch stops injecting
            # routing exploration for the rest of training.
            self.generator.train(gen_was)

    def compare_architectures(self, x1: torch.Tensor, x2: torch.Tensor) -> dict:
        gen_was = self.generator.training
        self.generator.eval()
        try:
            with torch.no_grad():
                m1 = self.generator(x1.to(self.device))
                m2 = self.generator(x2.to(self.device))
                dist = arch_distance(m1, m2)
                return {
                    "arch1": self.generate_architecture(x1),
                    "arch2": self.generate_architecture(x2),
                    "distance": dist.item(),
                }
        finally:
            self.generator.train(gen_was)

    def prototype_usage(self, x: torch.Tensor) -> dict:
        gen_was = self.generator.training
        self.generator.eval()
        try:
            with torch.no_grad():
                model = self.generator(x.to(self.device))
                pi = torch.softmax(model.pi_logits, dim=-1)
                assignments = pi.argmax(dim=-1)
                usage = {}
                for a in assignments.tolist():
                    usage[a] = usage.get(a, 0) + 1
                entropy = -(pi.mean(dim=0) * torch.log(pi.mean(dim=0) + 1e-8)).sum().item()
                return {"usage": usage, "entropy": entropy, "num_prototypes_used": len(usage)}
        finally:
            self.generator.train(gen_was)
