"""
profile_training_step.py - End-to-end PyTorch Profiler for AML Training Steps.

Measures:
- Total Step Time (CPU vs CUDA)
- Kernel vs Overhead Breakdown
- CUDA Memory Footprint & Allocation Count
- Top 15 Time-consuming CUDA Operations
"""

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))

import torch
import torch.nn.functional as F
from train import ConvAML_CIFAR_TrueDynamic

def profile_aml(batch_size=16, steps=10, warmup=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing model on {device} (Batch size = {batch_size})...")
    model = ConvAML_CIFAR_TrueDynamic(K=32, max_dim=64, max_layers=18, norm_mode="gn1").to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randn(batch_size, 3, 32, 32, device=device)
    target = torch.randint(0, 10, (batch_size,), device=device)

    print(f"[*] Warming up {warmup} steps...")
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        out, struct, pi, diag = model(x)
        loss = F.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    print(f"[*] Profiling {steps} steps with torch.profiler...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            out, struct, pi, diag = model(x)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()

    print("\n" + "=" * 80)
    print("TOP 15 CUDA KERNELS BY CUDA TOTAL TIME")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n" + "=" * 80)
    print("TOP 10 CPU OPERATIONS BY CPU TOTAL TIME (OVERHEAD ANALYSIS)")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    trace_path = os.path.join(_ROOT, "benchmarks", "aml_profile_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\n[*] Chrome trace exported to: {trace_path}")

if __name__ == "__main__":
    profile_aml()
