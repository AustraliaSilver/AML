"""Build the InstanceNorm fused-pipeline extension and deploy it as
mgn/accel/_Cin.cp312-win_amd64.pyd so `import mgn.accel._Cin` works natively.
The existing GN1 extension (_C) is left untouched."""
import os
import shutil
from torch.utils.cpp_extension import load

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "mgn", "csrc", "in")

mod = load(
    name="_Cin",
    sources=[
        os.path.join(SRC, "bindings_in.cpp"),
        os.path.join(SRC, "fused_pipeline_in.cu"),
        os.path.join(SRC, "fused_backward_in.cu"),
    ],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    verbose=True,
)
print("built:", mod.__file__)

built_pyd = mod.__file__
dst = os.path.join(ROOT, "mgn", "accel", "_Cin.cp312-win_amd64.pyd")
shutil.copy2(built_pyd, dst)
print("deployed to:", dst)

import mgn.accel._Cin as _Cin
print("native import OK ->", _Cin.__file__)
print("exports:", [x for x in dir(_Cin) if not x.startswith("_")])
