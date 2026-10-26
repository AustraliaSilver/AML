## 2023-10-26 - set_to_none=True optimization
**Learning:** Found several places where `zero_grad()` is missing `set_to_none=True`.
**Action:** Replace `zero_grad()` with `zero_grad(set_to_none=True)` in hot paths to improve performance.
