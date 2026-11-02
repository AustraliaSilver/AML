## 2026-10-16 - [Replace manual gating with torch.lerp]
**Learning:** PyTorch native `torch.lerp` executes significantly faster than manual linear interpolation math (`g * x + (1 - g) * y`) by avoiding memory overhead for creating intermediate tensors (like `(1.0 - g)` and the two products), and minimizing CUDA kernel launches. This is very relevant in deep looping gating models like dynamic routing/depth architectures in this codebase.
**Action:** Always prefer `torch.lerp(y, x, weight)` over `w * x + (1 - w) * y` for gating math in PyTorch models.
