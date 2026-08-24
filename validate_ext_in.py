"""validate_ext_in.py — numeric parity checks for the InstanceNorm extension.

1. Harness sanity: old GN1 fused (_C.fused_forward_and_save) vs eager GN1 ref.
2. New IN fused (_Cin.in_fused_forward_and_save) vs eager InstanceNorm ref.
3. Gradient parity through FusedPipelineFunctionIN vs autograd on the ref.
"""
import os, sys
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))
from train_cv_cifar_18 import get_prototype_layout

import mgn.accel._C as _Cold
import mgn.accel._Cin as _Cin
from mgn.accel.fused_autograd_in import FusedPipelineFunctionIN

torch.manual_seed(7)
dev = torch.device("cuda")

B, L, D, K, NC = 4, 18, 64, 8, 10
total_dim, offsets, layer_info, wo_off, bo_off = get_prototype_layout(L, D, NC)

x = torch.randn(B, 3, 32, 32, device=dev)
protos = torch.randn(K, total_dim, device=dev) * 0.05
pi = torch.softmax(torch.randn(B, K, device=dev), dim=-1)
lg = torch.sigmoid(torch.randn(B, L, device=dev))
dm = torch.rand(B, L, D, device=dev)
bm = torch.stack([torch.rand(B, L, device=dev),
                  torch.rand(B, L, device=dev)], dim=-1)

w_offsets = [offsets[i][0] for i in range(L)]
b_offsets = [offsets[i][2] for i in range(L)]
C_ins = [offsets[i][4] for i in range(L)]
C_outs = [D] * L
Hs, Ws = [32], [32]
for i in range(1, L):
    if i == 6 or i == 12:
        Hs.append(Hs[-1] // 2); Ws.append(Ws[-1] // 2)
    else:
        Hs.append(Hs[-1]); Ws.append(Ws[-1])


def ref_forward(x_, protos_, pi_, lg_, dm_, bm_, norm):
    h = x_.float()
    B_ = h.shape[0]
    for i in range(L):
        w0, w1, b0, b1, cin = offsets[i]
        cout = C_outs[i]; hd = Hs[i]; wd = Ws[i]
        Wm = torch.mm(pi_.to(torch.bfloat16),
                      protos_[:, w0:w1].to(torch.bfloat16)).float().view(B_, cout, cin, 3, 3)
        bx = torch.mm(pi_.to(torch.bfloat16),
                      protos_[:, b0:b1].to(torch.bfloat16)).float()
        hc = F.conv2d(h.reshape(1, B_ * cin, hd, wd).to(torch.bfloat16),
                      Wm.reshape(B_ * cout, cin, 3, 3).to(torch.bfloat16),
                      bx.reshape(-1).to(torch.bfloat16),
                      padding=1, groups=B_).float().view(B_, cout, hd, wd)
        if norm == "gn1":
            hn = F.group_norm(hc, 1)
        else:
            hn = F.instance_norm(hc)
        dmi = dm_[:, i].view(B_, D, 1, 1)
        gp = bm_[:, i, 0].view(-1, 1, 1, 1)
        gr = bm_[:, i, 1].view(-1, 1, 1, 1)
        lgv = lg_[:, i].view(-1, 1, 1, 1)
        hp = F.relu(hn) * dmi
        he = hp if i == 0 else h * dmi
        hr = F.relu(he + hn) * dmi
        ha = gp * hp + gr * hr
        h = ha if i == 0 else lgv * ha + (1 - lgv) * he
        if i == 5 or i == 11:
            h = F.max_pool2d(h, 2, 2)
    return h


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()


# ---- 1. harness sanity: old GN1 path --------------------------------------
with torch.no_grad():
    res_old = _Cold.fused_forward_and_save(
        x.float().contiguous(), protos.float().contiguous(), pi.float().contiguous(),
        lg.float().contiguous(), dm.float().contiguous(), bm.float().contiguous(),
        w_offsets, b_offsets, C_ins, C_outs, Hs, Ws, D)
ref_gn1 = ref_forward(x, protos, pi, lg, dm, bm, "gn1")
e_old = rel(res_old[0], ref_gn1)
print(f"[1] GN1 fused vs eager ref : rel_err {e_old:.5f}  "
      f"{'OK' if e_old < 5e-2 else 'FAIL'}")

# ---- 2. new IN path: forward parity ---------------------------------------
with torch.no_grad():
    res_in = _Cin.in_fused_forward_and_save(
        x.float().contiguous(), protos.float().contiguous(), pi.float().contiguous(),
        lg.float().contiguous(), dm.float().contiguous(), bm.float().contiguous(),
        w_offsets, b_offsets, C_ins, C_outs, Hs, Ws, D)
ref_in = ref_forward(x, protos, pi, lg, dm, bm, "in")
e_in = rel(res_in[0], ref_in)
print(f"[2] IN  fused vs eager ref : rel_err {e_in:.5f}  "
      f"{'OK' if e_in < 5e-2 else 'FAIL'}")

# ---- 3. gradient parity through the autograd wrapper ----------------------
xg = x.clone().requires_grad_(True)
pg = protos.clone().requires_grad_(True)
pig = pi.clone().requires_grad_(True)
lgg = lg.clone().requires_grad_(True)
dmg = dm.clone().requires_grad_(True)
bmg = bm.clone().requires_grad_(True)

out = FusedPipelineFunctionIN.apply(
    xg, pg, pig, lgg, dmg, bmg,
    w_offsets, b_offsets, C_ins, C_outs, Hs, Ws, D)
wloss = torch.randn_like(out)
(out * wloss).sum().backward()

xr = x.clone().requires_grad_(True)
pr = protos.clone().requires_grad_(True)
pir = pi.clone().requires_grad_(True)
lgr = lg.clone().requires_grad_(True)
dmr = dm.clone().requires_grad_(True)
bmr = bm.clone().requires_grad_(True)
out_r = ref_forward(xr, pr, pir, lgr, dmr, bmr, "in")
(out_r * wloss).sum().backward()

for name, a, b in [("protos", pg.grad, pr.grad), ("pi", pig.grad, pir.grad),
                   ("lg", lgg.grad, lgr.grad), ("dm", dmg.grad, dmr.grad),
                   ("bm", bmg.grad, bmr.grad)]:
    e = rel(a, b)
    print(f"[3] grad {name:7s}          : rel_err {e:.5f}  "
          f"{'OK' if e < 1e-1 else 'CHECK'}")
print("done")
