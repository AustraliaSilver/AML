"""debug_in_layer.py — unit tests isolating the IN post-conv kernels.

A. Single-layer backward correctness: kernel vs autograd (fp64) on the exact
   post-conv formula, for layer0 and deeper layers.
B. Divergence growth over depth: fused IN pipeline vs eager ref, per layer count.
"""
import os, sys
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "AML_Core_Code"))
from train_cv_cifar_18 import get_prototype_layout
import mgn.accel._Cin as _Cin

torch.manual_seed(11)
dev = torch.device("cuda")


def post_conv_torch(hc, hp, dm, bm, lg, is_layer0):
    """Reference implementation of one post-conv step (IN norm)."""
    hn = F.instance_norm(hc)
    B = hc.shape[0]
    dmi = dm.view(-1, dm.shape[-1], 1, 1)
    gp = bm[:, 0].view(-1, 1, 1, 1)
    gr = bm[:, 1].view(-1, 1, 1, 1)
    lgv = lg.view(-1, 1, 1, 1)
    hp_ = F.relu(hn) * dmi
    he = hp_ if is_layer0 else hp * dmi
    hr = F.relu(he + hn) * dmi
    ha = gp * hp_ + gr * hr
    return ha if is_layer0 else lgv * ha + (1 - lgv) * he


def unit_backward_test(is_layer0):
    B, C, H, W, D = 3, 8, 6, 6, 8
    hc64 = torch.randn(B, C, H, W, device=dev, dtype=torch.float64) * 2 + 1
    hp64 = torch.randn(B, C, H, W, device=dev, dtype=torch.float64)
    dm64 = torch.rand(B, D, device=dev, dtype=torch.float64)
    bm64 = torch.rand(B, 2, device=dev, dtype=torch.float64)
    lg64 = torch.rand(B, device=dev, dtype=torch.float64)

    hc = hc64.detach().requires_grad_(True)
    hp_ = hp64.detach().requires_grad_(True)
    dm = dm64.detach().requires_grad_(True)
    bm = bm64.detach().requires_grad_(True)
    lg = lg64.detach().requires_grad_(True)

    out = post_conv_torch(hc, hp_, dm, bm, lg, is_layer0)
    gout = torch.randn_like(out)
    out.backward(gout)

    res = _Cin.in_fused_post_conv_bwd(
        gout.float().contiguous(), hc64.float().contiguous(), hp64.float().contiguous(),
        dm64.float().contiguous(), bm64.float().contiguous(), lg64.float().contiguous(),
        torch.zeros(B, 2, device=dev), torch.zeros(B, device=dev),
        torch.zeros(B, D, device=dev),
        B, C, H * W, D, 1 if is_layer0 else 0
    )
    dh_conv_k, dh_prev_k = res[0], res[1]
    gbm_k, glg_k, gdm_k = None, None, None

    def rel(a, b):
        return ((a - b).norm() / b.norm().clamp(min=1e-12)).item()

    e_hc = rel(dh_conv_k.double(), hc.grad)
    e_hp = rel(dh_prev_k.double(), hp_.grad) if not is_layer0 else float("nan")

    # scalar/mask grads need re-run since buffers were consumed inside kernel call
    gbm_t = torch.zeros(B, 2, device=dev); glg_t = torch.zeros(B, device=dev)
    gdm_t = torch.zeros(B, D, device=dev)
    _Cin.in_fused_post_conv_bwd(
        gout.float().contiguous(), hc64.float().contiguous(), hp64.float().contiguous(),
        dm64.float().contiguous(), bm64.float().contiguous(), lg64.float().contiguous(),
        gbm_t, glg_t, gdm_t,
        B, C, H * W, D, 1 if is_layer0 else 0
    )
    e_dm = rel(gdm_t.double(), dm.grad)
    e_bm = rel(gbm_t.double(), bm.grad)
    e_lg = rel(glg_t.double(), lg.grad) if lg.grad is not None else float("nan")
    tag = f"layer0={is_layer0}"
    print(f"[A {tag}] dh_conv {e_hc:.2e} | dh_prev {e_hp:.2e} | "
          f"d_dm {e_dm:.2e} | d_bm {e_bm:.2e} | d_lg {e_lg:.2e}")


unit_backward_test(True)
unit_backward_test(False)


# ---- B. divergence growth over depth --------------------------------------
B, L, D, K, NC = 4, 18, 64, 8, 10
total_dim, offsets, _, _, _ = get_prototype_layout(L, D, NC)
x = torch.randn(B, 3, 32, 32, device=dev)
protos = (torch.randn(K, total_dim, device=dev) * 0.05)
pi = torch.softmax(torch.randn(B, K, device=dev), -1)
lg = torch.sigmoid(torch.randn(B, L, device=dev))
dm = torch.rand(B, L, D, device=dev)
bm = torch.stack([torch.rand(B, L, device=dev), torch.rand(B, L, device=dev)], -1)

w_offsets = [offsets[i][0] for i in range(L)]
b_offsets = [offsets[i][2] for i in range(L)]
C_ins = [offsets[i][4] for i in range(L)]
Hs, Ws = [32], [32]
for i in range(1, L):
    if i == 6 or i == 12:
        Hs.append(Hs[-1] // 2); Ws.append(Ws[-1] // 2)
    else:
        Hs.append(Hs[-1]); Ws.append(Ws[-1])


def ref_n(x_, protos_, pi_, lg_, dm_, bm_, n_layers):
    h = x_.float()
    for i in range(n_layers):
        w0, w1, b0, b1, cin = offsets[i]
        cout = D; hd = Hs[i]; wd = Ws[i]
        Wm = torch.mm(pi_.to(torch.bfloat16), protos_[:, w0:w1].to(torch.bfloat16)).float().view(B, cout, cin, 3, 3)
        bx = torch.mm(pi_.to(torch.bfloat16), protos_[:, b0:b1].to(torch.bfloat16)).float()
        hc = F.conv2d(h.reshape(1, B * cin, hd, wd).to(torch.bfloat16),
                      Wm.reshape(B * cout, cin, 3, 3).to(torch.bfloat16),
                      bx.reshape(-1).to(torch.bfloat16), padding=1, groups=B).float().view(B, cout, hd, wd)
        hn = F.instance_norm(hc)
        dmi = dm_[:, i].view(B, D, 1, 1)
        gp = bm_[:, i, 0].view(-1, 1, 1, 1); gr = bm_[:, i, 1].view(-1, 1, 1, 1)
        lgv = lg_[:, i].view(-1, 1, 1, 1)
        hp = F.relu(hn) * dmi
        he = hp if i == 0 else h * dmi
        hr = F.relu(he + hn) * dmi
        ha = gp * hp + gr * hr
        h = ha if i == 0 else lgv * ha + (1 - lgv) * he
        if i == 5 or i == 11:
            h = F.max_pool2d(h, 2, 2)
    return h


print("\n[B] depth sweep (fused IN vs eager IN):")
for nl in [1, 2, 4, 6, 7, 8, 12, 18]:
    with torch.no_grad():
        r = _Cin.in_fused_forward_and_save(
            x.float().contiguous(), protos.float().contiguous(), pi.float().contiguous(),
            lg.float().contiguous(), dm.float().contiguous(), bm.float().contiguous(),
            w_offsets[:nl], b_offsets[:nl], C_ins[:nl], [D] * nl, Hs[:nl], Ws[:nl], D)
        e = ((r[0] - ref_n(x, protos, pi, lg, dm, bm, nl)).norm()
             / ref_n(x, protos, pi, lg, dm, bm, nl).norm()).item()
    print(f"   L={nl:2d}: rel_err {e:.5f}")
