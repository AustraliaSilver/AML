## 2024-05-15 - [Model Init Optimization]
**Learning:** [The function `_valid_groups` determines group normalizations and is called during model instantiation for every block. While small, memoizing its calculation is safe and slightly improves instantiation speed.]
**Action:** [Memoize functions with purely determinist logic and small parameter combinations if called often during instantiation.]
