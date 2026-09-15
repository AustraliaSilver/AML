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
        preprocess_fn=None,
        stability_interval: int = 25,
        compile: bool = False,
        compile_mode: str = "default",
    ):
        """preprocess_fn: optional callable applied to every TRAIN x-batch
        (e.g. dataset-side horizontal flip for flat CIFAR). The library keeps
        no dataset-specific augmentation policy of its own; use torch RNG
        inside the fn so torch.manual_seed keeps the run reproducible.
        stability_interval: refresh cadence (in optimizer steps) of the
        slow-target snapshot used by the lambda_stability term. Refreshing
        every step deep-copies the whole generated model per batch for a
        ~1e-6-weighted term; a slow target is standard practice and
        removes ~95% of that memcpy overhead.
        compile: torch.compile() the generator and executor to fuse the
        thousands of per-iteration micro-kernels (the profiled bottleneck:
        ~2k empty/mul/select/add launches per batch). Falls back to eager
        with a warning if compilation fails. Off by default so recorded
        trajectories stay bitwise comparable; enable explicitly for speed.
        compile_mode: dynamo/inductor mode, e.g. "default" (safe, tolerates
        the partial last batch) or "reduce-overhead" (CUDA graphs; needs
        static shapes, i.e. drop_last loaders)."""
        self.preprocess_fn = preprocess_fn
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
        self.stability_interval = max(1, stability_interval)
        self._train_step = 0
        self.compiled = False
        if compile:
            import warnings
            try:
                # NOTE (Windows/torch 2.5): inductor is unusable on this
                # box -- even a 3-layer MLP fails with a WinError 183
                # TOCTOU race in codecache.write_atomic (os.rename cannot
                # overwrite). Use --compile on Linux; here it reliably
                # falls back to eager via the dry run below.
                # Fresh per-process cache dir (the supported knob is the
                # env var; assigning config.cache_dir does nothing).
                try:
                    import os
                    import tempfile
                    os.environ.setdefault(
                        "TORCHINDUCTOR_CACHE_DIR",
                        tempfile.mkdtemp(prefix="torchinductor_aml_"))
                    import torch._inductor.config as _ind_cfg
                    if hasattr(_ind_cfg, "force_disable_caches"):
                        _ind_cfg.force_disable_caches = True
                except Exception:
                    pass
                self.generator = torch.compile(self.generator, mode=compile_mode)
                self.executor = torch.compile(self.executor, mode=compile_mode)
                self._compile_dry_run(device)
                self.compiled = True
            except Exception as e:  # inductor/triton unavailable -> eager
                warnings.warn(f"MetaTrainer: torch.compile failed ({e}); "
                              f"continuing in eager mode.")
        # NOTE: self.generator/self.executor may now be OptimizedModule
        # wrappers; .parameters() still resolves to the original tensors,
        # so the optimizer below trains the same weights either way.

    def _compile_dry_run(self, device, batch_size: int = 64):
        """Force inductor compilation of forward+backward up front so a
        backend failure falls back to eager HERE (with identical weights)
        instead of mid-training. RNG state is saved/restored so the dry
        run does not perturb the training trajectory."""
        in_dim = getattr(self.generator, "input_dim", 784)
        out_dim = getattr(self.generator, "output_dim", 10)
        cpu_rng = torch.get_rng_state()
        cuda_rng = None
        if str(device).startswith("cuda") and torch.cuda.is_available():
            cuda_rng = torch.cuda.get_rng_state(device)
        try:
            self.generator.train()
            self.executor.train()
            x = torch.randn(batch_size, in_dim, device=device)
            y = torch.randint(0, out_dim, (batch_size,), device=device)
            model = self.generator(x, tau=1.0)
            logits = self.executor(x, model)
            logits.sum().backward()
            self.optimizer.zero_grad(set_to_none=True)
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, device)

    def train_epoch(self, dataloader: DataLoader, val_dataloader: DataLoader = None,
                    tau: float = 1.0, epoch_progress: float = 0.0,
                    task_loss_fn: nn.Module = None) -> dict:
        self.generator.train()
        self.executor.train()
        # GPU-side running sums: a single .item() per metric at epoch end
        # instead of 5 device syncs per batch. Values logged are identical;
        # only the sync points move.
        run_sums = {"loss": 0.0, "task_loss": 0.0, "resource_cost": 0.0,
                    "acc": 0.0, "aml_objective": 0.0}
        num_batches = 0
        val_iter = iter(val_dataloader) if val_dataloader is not None else None
        use_stability = self.meta_loss.lambda_stability > 0

        for batch in dataloader:
            if len(batch) == 3:
                x_batch, y_batch, task_ids = batch
            else:
                x_batch, y_batch = batch
                task_ids = None
            x_batch = x_batch.to(self.device, non_blocking=True)
            if self.preprocess_fn is not None and self.generator.training:
                x_batch = self.preprocess_fn(x_batch)
            y_batch = y_batch.to(self.device, non_blocking=True)

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
                
                val_x = val_x.to(self.device, non_blocking=True)
                val_y = val_y.to(self.device, non_blocking=True)
                
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
                kwargs["task_ids"] = task_ids[:n_contrast].to(self.device, non_blocking=True)

            metrics = self.meta_loss(logits_train, y_train, logits_val, y_val, model, **kwargs)
            total_loss = metrics["loss"]

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
            self.optimizer.step()
            self._train_step += 1

            # Slow-target refresh: the stability snapshot is a full deep copy
            # of the generated model, so rebuild it every stability_interval
            # steps instead of every batch. Skipped entirely when the
            # stability term is off (it was still cloned per batch before).
            if use_stability and (self._prev_model is None
                                  or self._train_step % self.stability_interval == 0):
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

            # Detached GPU-side accumulation; synced once at epoch end below.
            run_sums["loss"] = run_sums["loss"] + total_loss.detach()
            _tl = metrics["task_loss"]
            run_sums["task_loss"] = run_sums["task_loss"] + (_tl.detach() if torch.is_tensor(_tl) else _tl)
            _rc = metrics["resource_cost"]
            run_sums["resource_cost"] = run_sums["resource_cost"] + (_rc.detach() if torch.is_tensor(_rc) else _rc)
            run_sums["acc"] = run_sums["acc"] + acc.detach()
            _ao = metrics.get("aml_objective", 0.0)
            run_sums["aml_objective"] = run_sums["aml_objective"] + (_ao.detach() if torch.is_tensor(_ao) else _ao)
            num_batches += 1

        if num_batches == 0:
            return {k: 0.0 for k in ("loss", "task_loss", "resource_cost", "acc", "aml_objective")}
        avg = {}
        for k, v in run_sums.items():
            v = v.item() if torch.is_tensor(v) else v
            avg[k] = v / num_batches
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
        gate_floor_start: float = 0.0,
        gate_floor_end: float = 0.0,
        freeze_arch_at: float = None,
    ):
        dataset = TensorDataset(x, y)
        _pin = str(self.device).startswith("cuda")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            pin_memory=_pin)

        val_loader = None
        if val_x is not None and val_y is not None:
            val_dataset = TensorDataset(val_x, val_y)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,
                                    pin_memory=_pin)

        self._prev_model = None
        self._train_step = 0
        # Depth keep-alive: linearly anneal the generator's gate_floor
        # (see ModelGenerator docs) so deep paths stay gradient-alive while
        # the efficiency penalty ramps. Defaults 0->0 = original behaviour.
        # NOTE: compiled generators re-specialize when the attribute
        # changes; keep the schedule coarse under torch.compile.
        _gen = getattr(self.generator, "_orig_mod", self.generator)

        import sys
        print_every = 1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)
        # Two-phase (DARTS-style): search the architecture for the first
        # fraction of training, then freeze it and finetune weights only.
        # Single-phase joint search overfits the structure to train noise
        # (measured -11pt test on CIFAR-10 fast/seed0 vs frozen arch at
        # identical train fit). None = original single-phase behaviour.
        arch_frozen = False
        freeze_ep = (int(freeze_arch_at * epochs) if freeze_arch_at is not None
                     and 0.0 < freeze_arch_at < 1.0 else None)
        for epoch in range(epochs):
            epoch_progress = epoch / max(1, epochs - 1)
            if freeze_ep is not None and not arch_frozen and epoch >= freeze_ep:
                _g = getattr(self.generator, "_orig_mod", self.generator)
                n_frozen = 0
                for mod_name in ("prototype_selector", "structure_delta",
                                 "block_mode_head"):
                    mod = getattr(_g, mod_name, None)
                    if mod is not None:
                        for p in mod.parameters():
                            p.requires_grad_(False)
                            n_frozen += 1
                arch_frozen = True
                # Drop frozen tensors from the optimizer: some foreach
                # kernels do not accept a mix of None/non-None grads.
                for group in self.optimizer.param_groups:
                    group["params"] = [p for p in group["params"]
                                       if p.requires_grad]
                print(f"  [MetaTrainer] arch frozen at epoch {epoch+1}/{epochs} "
                      f"({n_frozen} tensors) -- finetune weights only")
                sys.stdout.flush()
            if gate_floor_start != 0.0 or gate_floor_end != 0.0:
                if hasattr(_gen, "gate_floor"):
                    _gen.gate_floor = (gate_floor_start
                                       + (gate_floor_end - gate_floor_start)
                                       * epoch_progress)
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
