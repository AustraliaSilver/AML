"""
verify_env.py - Environment verification and diagnostics for AML.

Checks:
1. Python version and platform
2. PyTorch installation & CUDA runtime support
3. GPU device specifications & compute capability
4. CUDA compiler (nvcc) & host C++ compiler (MSVC / gcc)
5. Native CUDA extension modules (_C, _Cin) status and export symbols
6. End-to-end forward & backward smoke check on GPU
"""

import sys
import shutil
import subprocess

def print_section(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def check_environment():
    all_ok = True
    print_section("1. Python & System Information")
    print(f"[*] Python Version : {sys.version.split()[0]}")
    print(f"[*] Platform       : {sys.platform}")
    if sys.version_info < (3, 9):
        print("[-] Warning: Python 3.9+ is recommended.")
        all_ok = False
    else:
        print("[+] Python version OK")

    print_section("2. PyTorch & CUDA Information")
    try:
        import torch
        print(f"[*] PyTorch Version : {torch.__version__}")
        print(f"[*] CUDA Available  : {torch.cuda.is_available()}")
        if not torch.cuda.is_available():
            print("[-] Error: CUDA is not available in PyTorch.")
            return False
        
        device_name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[*] Primary GPU     : {device_name}")
        print(f"[*] Compute Cap     : {cap[0]}.{cap[1]} (SM {cap[0]}{cap[1]})")
        print(f"[*] Total VRAM      : {vram:.2f} GB")
        print(f"[*] BF16 Supported  : {torch.cuda.is_bf16_supported()}")
        print("[+] PyTorch CUDA environment OK")
    except ImportError:
        print("[-] Error: PyTorch is not installed.")
        return False

    print_section("3. Compiler Toolchain (NVCC & C++)")
    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        try:
            nvcc_ver = subprocess.check_output(["nvcc", "--version"], text=True)
            for line in nvcc_ver.splitlines():
                if "release" in line:
                    print(f"[*] NVCC Compiler  : {line.strip()} ({nvcc_path})")
            print("[+] NVCC found OK")
        except Exception as e:
            print(f"[-] NVCC execution failed: {e}")
            all_ok = False
    else:
        print("[-] Warning: nvcc not found in PATH.")
        all_ok = False

    if sys.platform == "win32":
        cl_path = shutil.which("cl")
        if cl_path:
            print(f"[*] MSVC Compiler  : {cl_path}")
            print("[+] MSVC found OK")
        else:
            print("[-] Notice: cl.exe not in PATH (build_ext loads vcvars64.bat automatically).")

    print_section("4. Native CUDA Acceleration Extensions")
    try:
        from mgn.accel import _C
        exports_c = [k for k in dir(_C) if not k.startswith("__")]
        print(f"[+] Extension '_C' loaded successfully ({len(exports_c)} exports):")
        print(f"    {exports_c}")
    except Exception as e:
        print(f"[-] Error loading '_C': {e}")
        all_ok = False

    try:
        from mgn.accel import _Cin
        exports_cin = [k for k in dir(_Cin) if not k.startswith("__")]
        print(f"[+] Extension '_Cin' loaded successfully ({len(exports_cin)} exports):")
        print(f"    {exports_cin}")
    except Exception as e:
        print(f"[-] Error loading '_Cin': {e}")
        all_ok = False

    print_section("5. End-to-End Native GPU Kernel Smoke Test")
    try:
        import torch
        from mgn.accel import _C
        B, C_in, C_out, H, W = 2, 16, 32, 16, 16
        x = torch.randn(B, C_in, H, W, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(B, C_out * C_in * 9, device="cuda", dtype=torch.bfloat16)
        g = torch.randn(B, C_out, H, W, device="cuda", dtype=torch.float32)

        # 1. Conv backward test
        din, dw, db = _C.dynamic_conv2d_bwd(g, x, w, B, C_in, C_out, H, W)
        assert not torch.isnan(din).any(), "NaN in din"
        print("[+] Dynamic Conv2D backward kernel: OK")

        # 2. GroupNorm backward test
        raw = torch.randn(B, C_out, H, W, device="cuda")
        mean = raw.mean(dim=(1, 2, 3))
        rstd = torch.rsqrt(raw.var(dim=(1, 2, 3), unbiased=False) + 1e-5)
        dgn = _C.group_norm1_bwd(g, raw, mean, rstd, B, C_out, H, W)
        assert not torch.isnan(dgn).any(), "NaN in dgn"
        print("[+] GroupNorm1 backward kernel: OK")

        # 3. Batched GEMM & Tensor Core test
        A = torch.randn(4, 32, 32, device="cuda")
        B_mat = torch.randn(4, 32, 32, device="cuda")
        c_tiled = _C.batched_hyper_linear(A, B_mat, None)
        c_tc = _C.batched_hyper_linear_tc(A, B_mat, None)
        assert not torch.isnan(c_tiled).any(), "NaN in c_tiled"
        assert not torch.isnan(c_tc).any(), "NaN in c_tc"
        print("[+] Tiled GEMM & Tensor Core WMMA GEMM: OK")

    except Exception as e:
        print(f"[-] Smoke test failed: {e}")
        all_ok = False

    print_section("Summary Report")
    if all_ok:
        print("[SUCCESS] All environment components and native CUDA kernels are healthy and verified!")
    else:
        print("[WARNING] Some checks reported warnings or errors. Check the details above.")
    return all_ok

if __name__ == "__main__":
    ok = check_environment()
    sys.exit(0 if ok else 1)
