"""mgn._selfcheck -- the `aml-selfcheck` console script.

Answers one question on a machine you have never used before: does
`pip install aml-framework` actually give you a working framework?

Checks, in order:
  1. Distribution metadata (version) and interpreter / platform.
  2. torch + torchvision import, CUDA availability, the device in use.
  3. Whether the OPTIONAL fused kernels (mgn.accel._C / _Cin) are present.
     Their absence is NOT a failure: the wheel on PyPI is pure Python and
     mgn.accel falls back to the equivalent PyTorch GroupedConv path.
  4. A real forward + backward through the packaged hybrid supernet on random
     data (no dataset download), asserting a finite loss and real gradients.
  5. A real forward through the packaged hts_lib continual hypernetwork.

Exit code 0 when 1-2-4-5 pass (kernels stay optional), 1 otherwise. The
assertions are the point: an install that imports but cannot run a backward
pass is broken, and that is exactly the failure a fresh machine surfaces.
"""

from __future__ import annotations

import platform
import sys

__all__ = ["main"]

OK = "[+]"
BAD = "[-]"
INFO = "[*]"


def _dist_version() -> str:
    try:
        from importlib.metadata import version

        return version("aml-framework")
    except Exception:
        return "unknown (running from a source checkout, not an installed wheel)"


def main(argv=None) -> int:
    failures: list[str] = []

    print("=" * 70)
    print(f" aml-framework self-check  (version {_dist_version()})")
    print("=" * 70)

    print(f"{INFO} python  : {sys.version.split()[0]} on {platform.platform()}")
    if sys.version_info < (3, 10):
        print(f"{BAD} python >=3.10 is required (requires-python in pyproject.toml)")
        failures.append("python<3.10")

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - only on a broken install
        print(f"{BAD} torch import failed: {exc}")
        print(f"{BAD} FAILED: torch")
        return 1

    print(f"{INFO} torch   : {torch.__version__}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        print(f"{INFO} device  : cuda:{torch.cuda.current_device()} "
              f"({torch.cuda.get_device_name(0)}, sm {cap[0]}{cap[1]})")
    else:
        print(f"{INFO} device  : cpu  (CUDA not available -- the framework runs, "
              f"just without a GPU)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- optional fused kernels -------------------------------------------------
    for name in ("_C", "_Cin"):
        try:
            mod = __import__(f"mgn.accel.{name}", fromlist=[name])
            print(f"{OK} mgn.accel.{name} : loaded from {mod.__file__}")
        except Exception as exc:
            print(f"{INFO} mgn.accel.{name} : not built ({type(exc).__name__}); "
                  f"optional, using the PyTorch fallback")

    # --- packaged supernet: forward + backward ----------------------------------
    try:
        from mgn.blocks.hybrid_18 import ConvAML_18Layer_Hybrid

        torch.manual_seed(0)
        model = ConvAML_18Layer_Hybrid(K=4, max_layers=3, max_dim=16,
                                       num_classes=5).to(device)
        model.train()
        x = torch.randn(2, 3, 32, 32, device=device)
        out, struct, pi, _ = model(x)
        y = torch.randint(0, 5, (2,), device=device)
        loss = torch.nn.functional.cross_entropy(out.float(), y)
        assert torch.isfinite(loss), f"loss is not finite: {loss}"
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads, "backward produced no gradients"
        assert any(g.abs().sum().item() > 0 for g in grads), \
            "every gradient is identically zero"
        active = struct.layer_gates.detach().float().mean().item()
        print(f"{OK} mgn supernet : forward {tuple(out.shape)} -> loss "
              f"{loss.item():.4f}; backward ok "
              f"({len(grads)} tensors got gradient, mean gate {active:.3f})")
    except Exception as exc:
        print(f"{BAD} mgn supernet : {type(exc).__name__}: {exc}")
        failures.append("mgn")

    # --- packaged continual hypernetwork ----------------------------------------
    try:
        from hts_lib.models.hypernetwork import HtSContinualModel

        torch.manual_seed(0)
        hts = HtSContinualModel(d1=24, d2=48, d3=96, K=8,
                                num_classes=10).to(device)
        with torch.no_grad():
            out_h, alpha = hts(torch.randn(2, 3, 32, 32, device=device),
                               task="cifar")
        assert torch.isfinite(out_h).all(), "hts output is not finite"
        print(f"{OK} hts_lib      : forward {tuple(out_h.shape)} ok "
              f"(alpha {tuple(alpha.shape)})")
    except Exception as exc:
        print(f"{BAD} hts_lib      : {type(exc).__name__}: {exc}")
        failures.append("hts_lib")

    print("=" * 70)
    if failures:
        print(f"{BAD} FAILED: {', '.join(failures)}")
        return 1
    print(f"{OK} OK: the framework imports, forwards and backprops. "
          f"Fused kernels are optional.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
