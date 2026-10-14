"""
benchmarks/benchmark_cl_aml_vs_hts.py
======================================
Unified Dual-Engine Benchmark comparing:
  1. AML Supernet (Architecture Meta-Learning with ProtoIsolator & Snapshot Retrieval)
  2. HtS Dual-Residual Hypernetwork (with Native Dynamic Conv2D, Tensor Cores WMMA & HtSBasisIsolator)

Evaluates on the standard 3-task continual sequence:
  Task 1: MNIST (10 classes)
  Task 2: CIFAR-10 (10 classes)
  Task 3: Fashion-MNIST (10 classes)

Reports:
  - Accuracy Matrix A[i,j] for both engines
  - Backward Transfer (BWT) & Average Forgetting
  - Wall-clock Training & Evaluation Latency
  - Parameter Efficiency & Memory Footprint
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "AML_Core_Code"))

from train import ConvAML_CIFAR_TrueDynamic, set_seed
from train_continual_aml import (
    get_task_data,
    ProtoIsolator,
    collect_structure_slot,
    evaluate as evaluate_aml
)
from hts_lib.models.hypernetwork import HtSContinualModel
from hts_lib.training.trainer import ContinualTrainer


def run_benchmark(epochs_aml=(2, 2, 2), epochs_hts=(2, 2, 2), batch_size=128, steps_per_epoch=None, eval_cap=None, aml_k=12, aml_dim=48, device="cuda"):
    set_seed(42)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    data_root = os.path.join(ROOT, "data")
    tasks = ["mnist", "cifar10", "fashion"]
    hts_task_names = ["mnist", "cifar", "fashion"]
    T = len(tasks)

    print("\n" + "=" * 90)
    print(f" UNIFIED CONTINUAL LEARNING BENCHMARK: AML vs HtS ({device})")
    print(" Protocol: 3-Task Sequence (MNIST -> CIFAR-10 -> Fashion-MNIST)")
    print(f" Schedule: AML epochs={epochs_aml} | HtS epochs={epochs_hts}")
    print(f" AML Config: K={aml_k} prototypes | max_dim={aml_dim}")
    if steps_per_epoch:
        print(f" Train Cap: {steps_per_epoch} batches per epoch")
    if eval_cap:
        print(f" Eval Cap: {eval_cap} batches per evaluation")
    print("=" * 90)

    # -------------------------------------------------------------
    # Load DataLoaders
    # -------------------------------------------------------------
    loaders = {}
    test_loaders = {}
    for t in tasks:
        tr, te = get_task_data(t, data_root, workers=0)
        loaders[t] = tr
        if eval_cap:
            class LimitedLoader:
                def __init__(self, loader, cap):
                    self.loader = loader
                    self.cap = cap
                def __iter__(self):
                    count = 0
                    for item in self.loader:
                        yield item
                        count += 1
                        if count >= self.cap:
                            break
                def __len__(self):
                    return min(len(self.loader), self.cap)
            test_loaders[t] = LimitedLoader(te, eval_cap)
        else:
            test_loaders[t] = te

    amp_dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else (
        torch.float16 if device.type == "cuda" else None)

    # =============================================================
    # 1. EVALUATE AML ENGINE
    # =============================================================
    print("\n" + "#" * 90)
    print(" [1/2] RUNNING AML SUPERNET (ProtoIsolator + Native Fused CUDA V2)")
    print("#" * 90)

    aml_model = ConvAML_CIFAR_TrueDynamic(K=aml_k, max_dim=aml_dim, max_layers=18, norm_mode="gn1").to(device)
    aml_isolator = ProtoIsolator(aml_model)
    dom_head = nn.Linear(1024, T).to(device)
    slots = [None] * T
    dom_snaps = [None] * T

    A_aml_live = [[None] * T for _ in range(T)]
    A_aml_method = [[None] * T for _ in range(T)]

    aml_start_time = time.time()

    for i, t_name in enumerate(tasks):
        print(f"\n---> [AML] Task {i+1}/{T}: {t_name.upper()} ({epochs_aml[i]} epochs)")
        s, e = aml_isolator.begin_task(i, T, reinit=False)
        pi_mask = torch.zeros(aml_model.K, dtype=torch.bool, device=device)
        pi_mask[s:e] = True

        params = [
            {"params": aml_model.vision_encoder.parameters(), "lr": 3e-4, "weight_decay": 1e-4},
            {"params": aml_model.prototype_selector.parameters(), "lr": 1e-3, "weight_decay": 1e-4},
            {"params": [aml_model.proto_weights, aml_model.structure_prototypes, aml_model.block_mode_prototypes], "lr": 1.5e-3, "weight_decay": 1e-5},
        ]
        opt = torch.optim.AdamW(params)
        # The routing head (whether it is a learned linear or a centroid
        # prototype) must also be optimised, otherwise autonomous routing
        # never trains on this benchmark.
        opt_dom = torch.optim.AdamW(dom_head.parameters(), lr=1e-3, weight_decay=0.0)

        t_task_start = time.time()
        for ep in range(1, epochs_aml[i] + 1):
            aml_model.train()
            batch_count = 0
            for x, y in loaders[t_name]:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                opt_dom.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=amp_dtype is not None, dtype=amp_dtype or torch.bfloat16):
                    out, _, pi, diag = aml_model(x, tau=0.8, pi_mask=pi_mask)
                    loss = F.cross_entropy(out, y)

                    if e > s:
                        p_act = aml_model.proto_weights[s:e]
                        p_n = F.normalize(p_act, dim=-1)
                        sim = p_n @ p_n.t()
                        eye = torch.eye(sim.shape[0], device=device)
                        loss = loss + 0.05 * ((sim - eye) ** 2).mean()

                    if diag is not None and "z" in diag:
                        z_norm = F.normalize(diag["z"].float(), dim=1)
                        dom_logits = dom_head(z_norm)
                        valid = i + 1
                        loss = loss + 0.3 * F.cross_entropy(dom_logits[:, :valid], torch.full((x.shape[0],), i, device=device, dtype=torch.long))

                loss.backward()
                aml_isolator.mask_grad()
                torch.nn.utils.clip_grad_norm_(aml_model.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(dom_head.parameters(), 5.0)
                opt.step()
                opt_dom.step()
                aml_isolator.restore()

                batch_count += 1
                if steps_per_epoch and batch_count >= steps_per_epoch:
                    break

        print(f"      Task {t_name} trained in {time.time() - t_task_start:.1f}s")

        # Snapshot for method protocol
        dom_snaps[i] = {
            "enc": {k: v.detach().clone() for k, v in aml_model.vision_encoder.state_dict().items()},
            "sel": {k: v.detach().clone() for k, v in aml_model.prototype_selector.state_dict().items()},
            "protos": aml_model.proto_weights.detach().clone().cpu(),
        }
        slots[i] = collect_structure_slot(aml_model, test_loaders[t_name], device, amp_dtype)

        # Evaluate after stage i
        print(f"      Evaluating AML after stage {i+1}...")
        for j in range(i + 1):
            s_j = round(j * aml_model.K / T)
            e_j = round((j + 1) * aml_model.K / T)
            pi_mask_j = torch.zeros(aml_model.K, dtype=torch.bool, device=device)
            pi_mask_j[s_j:e_j] = True

            acc_live, _ = evaluate_aml(aml_model, test_loaders[tasks[j]], device, amp_dtype=amp_dtype, pi_mask=pi_mask_j)
            acc_meth, _ = evaluate_aml(
                aml_model, test_loaders[tasks[j]], device, amp_dtype=amp_dtype,
                dom_head=dom_head, slots=slots[:i+1], snaps=dom_snaps[:i+1],
                route_tau=0.6, dom_truth=j, pi_mask=pi_mask_j
            )
            A_aml_live[i][j] = acc_live * 100.0
            A_aml_method[i][j] = acc_meth * 100.0

        print("      [Live Accs]   : " + ", ".join(f"{tasks[j]}: {A_aml_live[i][j]:.2f}%" for j in range(i+1)))
        print("      [Method Accs] : " + ", ".join(f"{tasks[j]}: {A_aml_method[i][j]:.2f}%" for j in range(i+1)))

    aml_total_time = time.time() - aml_start_time

    # =============================================================
    # 2. EVALUATE HtS ENGINE
    # =============================================================
    print("\n" + "#" * 90)
    print(" [2/2] RUNNING HtS DUAL-RESIDUAL (Tensor Core WMMA + HtSBasisIsolator No-Replay)")
    print("#" * 90)

    hts_model = HtSContinualModel(d1=24, d2=48, d3=96, K=8, num_classes=10).to(device)
    hts_trainer = ContinualTrainer(
        hts_model, device=device, lr=1.5e-3,
        lambda_ent=0.015, lambda_use=0.10, lambda_ort=0.05,
        enable_basis_isolation=True
    )

    A_hts = [[None] * T for _ in range(T)]
    hts_start_time = time.time()

    for i, t_name in enumerate(tasks):
        hts_t = hts_task_names[i]
        print(f"\n---> [HtS] Task {i+1}/{T}: {t_name.upper()} ({epochs_hts[i]} epochs)")
        t_task_start = time.time()

        # Custom loader if capped
        train_iter = loaders[t_name]
        if steps_per_epoch:
            # Wrap limited batches
            class LimitedLoader:
                def __init__(self, loader, cap):
                    self.loader = loader
                    self.cap = cap
                def __iter__(self):
                    count = 0
                    for item in self.loader:
                        yield item
                        count += 1
                        if count >= self.cap:
                            break
            train_iter = LimitedLoader(loaders[t_name], steps_per_epoch)

        hts_trainer.train_task(
            current_task=hts_t,
            train_loader=train_iter,
            epochs=epochs_hts[i],
            replay_loaders={},
            task_idx=i,
            num_tasks=T,
            use_tc=True
        )
        print(f"      Task {t_name} trained in {time.time() - t_task_start:.1f}s")

        # Evaluate after stage i
        print(f"      Evaluating HtS after stage {i+1}...")
        hts_model.eval()
        for j in range(i + 1):
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in test_loaders[tasks[j]]:
                    x, y = x.to(device), y.to(device)
                    out, _ = hts_model(x, task=hts_task_names[j], use_tc=True)
                    correct += (out.argmax(-1) == y).sum().item()
                    total += y.numel()
            acc = correct / total * 100.0
            A_hts[i][j] = acc

        print("      [HtS Accs]    : " + ", ".join(f"{tasks[j]}: {A_hts[i][j]:.2f}%" for j in range(i+1)))

    hts_total_time = time.time() - hts_start_time

    # =============================================================
    # METRICS SUMMARY CALCULATION
    # =============================================================
    def calc_metrics(A_mat):
        # A_mat is list of list: A_mat[i][j]
        avg_final = sum(A_mat[T-1][j] for j in range(T)) / T
        bwt = sum(A_mat[T-1][j] - A_mat[j][j] for j in range(T - 1)) / (T - 1)
        forg_list = []
        for j in range(T - 1):
            peak = max(A_mat[t][j] for t in range(j, T - 1))
            forg_list.append(peak - A_mat[T-1][j])
        avg_forg = sum(forg_list) / len(forg_list)
        return avg_final, bwt, avg_forg

    m_aml_live, bwt_aml_live, forg_aml_live = calc_metrics(A_aml_live)
    m_aml_meth, bwt_aml_meth, forg_aml_meth = calc_metrics(A_aml_method)
    m_hts, bwt_hts, forg_hts = calc_metrics(A_hts)

    # Parameter counts
    p_aml = sum(p.numel() for p in aml_model.parameters())
    p_hts = sum(p.numel() for p in hts_model.parameters())

    print("\n" + "=" * 90)
    print("                     FINAL CONTINUAL BENCHMARK RESULTS")
    print("=" * 90)

    print("\n[1] ACCURACY MATRIX: AML LIVE (Standard CL - Current Weights Only)")
    for i in range(T):
        row = " | ".join(f"{A_aml_live[i][j]:6.2f}%" if j <= i else "   ---  " for j in range(T))
        print(f"    After Stage {i+1} ({tasks[i]:<10}) : {row}")

    print("\n[2] ACCURACY MATRIX: AML METHOD (Test-time Snapshot Retrieval)")
    for i in range(T):
        row = " | ".join(f"{A_aml_method[i][j]:6.2f}%" if j <= i else "   ---  " for j in range(T))
        print(f"    After Stage {i+1} ({tasks[i]:<10}) : {row}")

    print("\n[3] ACCURACY MATRIX: HtS (Dual-Residual Basis + Tensor Cores + HtSBasisIsolator)")
    for i in range(T):
        row = " | ".join(f"{A_hts[i][j]:6.2f}%" if j <= i else "   ---  " for j in range(T))
        print(f"    After Stage {i+1} ({tasks[i]:<10}) : {row}")

    print("\n" + "-" * 90)
    print(f"{'Engine / Method':<30} | {'Final ACC':<11} | {'BWT':<11} | {'Forgetting':<12} | {'Time (s)':<10} | {'Params':<10}")
    print("-" * 90)
    print(f"{'AML (Live Weights - No Replay)':<30} | {m_aml_live:>7.2f}%   | {bwt_aml_live:>+7.2f}%   | {forg_aml_live:>8.2f}%    | {aml_total_time:>8.1f}s | {p_aml:>10,}")
    print(f"{'AML (Method - Snapshot Retr.)':<30} | {m_aml_meth:>7.2f}%   | {bwt_aml_meth:>+7.2f}%   | {forg_aml_meth:>8.2f}%    | {aml_total_time:>8.1f}s | {p_aml:>10,}")
    print(f"{'HtS (No Replay + Tensor Cores)':<30} | {m_hts:>7.2f}%   | {bwt_hts:>+7.2f}%   | {forg_hts:>8.2f}%    | {hts_total_time:>8.1f}s | {p_hts:>10,}")
    print("=" * 90)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual Continual Learning Benchmark: AML vs HtS")
    parser.add_argument("--epochs", type=str, default="2,2,2", help="Epochs per task, e.g. 2,2,2")
    parser.add_argument("--steps", type=int, default=None, help="Cap batches per epoch for quick benchmarking")
    parser.add_argument("--eval-cap", type=int, default=None, help="Cap batches per evaluation")
    parser.add_argument("--aml-k", type=int, default=12, help="Number of prototypes K in AML (default: 12)")
    parser.add_argument("--aml-dim", type=int, default=48, help="Max dimension in AML (default: 48)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    ep_list = tuple(int(x) for x in args.epochs.split(","))
    run_benchmark(epochs_aml=ep_list, epochs_hts=ep_list, steps_per_epoch=args.steps, eval_cap=args.eval_cap,
                  aml_k=args.aml_k, aml_dim=args.aml_dim, device=args.device)
