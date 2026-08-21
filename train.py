import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AML_Core_Code'))
from train_cv_cifar_18 import AcceleratedCVNetwork, get_prototype_layout

# Use mgn from current directory (where we compiled it)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mgn.architecture import ModelStructure, GeneratedModel
from mgn.losses import MetaLoss

class ConvAML_CIFAR_TrueDynamic(nn.Module):
    def __init__(self, K=32, latent_dim=128, max_layers=18, max_dim=64):
        super().__init__()
        self.K = K
        self.max_layers = max_layers
        self.max_dim = max_dim

        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.ReLU()
        )
        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, K)
        )
        
        self.structure_delta = nn.Linear(latent_dim, max_layers + (max_layers * max_dim))
        self.block_mode_head = nn.Linear(latent_dim, max_layers * 3)
        nn.init.constant_(self.structure_delta.weight, 0.0)
        nn.init.constant_(self.structure_delta.bias, 0.0)
        nn.init.constant_(self.block_mode_head.weight, 0.0)
        nn.init.constant_(self.block_mode_head.bias, 0.0)
        with torch.no_grad():
            self.block_mode_head.bias.data.view(max_layers, 3)[:, 1] = 2.0
            self.block_mode_head.bias.data.view(max_layers, 3)[:, 2] = 1.0

        weight_dim = 0
        for i in range(max_layers):
            in_c = 3 if i == 0 else max_dim
            weight_dim += max_dim * in_c * 9 + max_dim
            attn_dim = max_dim // 4
            weight_dim += (3 * attn_dim) * max_dim + 3 * attn_dim
            weight_dim += max_dim * attn_dim + max_dim
        weight_dim += (16 * max_dim) * 10 + 10
        self._init_prototypes()
        
        self.exec_ = AcceleratedCVNetwork(max_layers=max_layers, max_dim=max_dim, num_classes=10)

    def _init_prototypes(self):
        import math
        import torch.nn as nn
        
        weight_dim, offsets, layer_info, wo_offset, bo_offset = get_prototype_layout(self.max_layers, self.max_dim, 10)
        self.proto_weights = nn.Parameter(torch.zeros(self.K, weight_dim))
        
        # We can just use a global kaiming uniform for simplicity, or zero out biases if we want to be strict.
        # For simplicity and correctness with the new layout, a global init is sufficient and less error-prone.
        nn.init.kaiming_uniform_(self.proto_weights, a=math.sqrt(5))

    def caf_attention(self, features):
        return torch.softmax(features, dim=-1).clamp(min=1e-8)

    def forward(self, x, tau=1.0):
        B = x.shape[0]; device = x.device
        z = self.vision_encoder(x)
        pi = self.caf_attention(self.prototype_selector(z))
        
        delta = self.structure_delta(z)
        layer_logits = delta[:, :self.max_layers] + 2.0
        layer_soft = torch.sigmoid(layer_logits)
        if self.training:
            layer_soft = layer_soft + 0.1 * torch.randn_like(layer_soft)
        layer_soft = torch.clamp(layer_soft, 1e-4, 1.0)
        layer_gates = (layer_soft > 0.5).float().detach() + layer_soft - layer_soft.detach()
        
        dim_logits = delta[:, self.max_layers:].view(B, self.max_layers, self.max_dim)
        d_logit = dim_logits.mean(dim=-1, keepdim=True) + 2.0
        dim_fractions = torch.sigmoid(d_logit)
        
        d_lengths = torch.clamp(dim_fractions * self.max_dim, min=24.0)
        arange = torch.arange(self.max_dim, device=device).float().view(1, 1, self.max_dim)
        dim_masks = torch.sigmoid(10.0 * (d_lengths - arange))

        bm_logits = self.block_mode_head(z).view(B, self.max_layers, 3)
        block_mode = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1)
            
        dim_weights = d_lengths / self.max_dim
        
        struct = ModelStructure(
            layer_gates=layer_gates,
            dim_weights=dim_weights,
            residual=torch.zeros(B, 1, device=device),
            dim_masks=dim_masks,
            block_mode=block_mode,
            use_conv=True, in_channels=3, max_layers=self.max_layers, max_dim=self.max_dim,
            dim_choices=(self.max_dim,), output_dim=10
        )
        
        out = self.exec_(x, self.proto_weights, pi, struct)
        diag = {
            "layer_soft": layer_soft,
            "bm_logits": bm_logits,
            "d_lengths": d_lengths,
        }
        return out, struct, pi, diag

def _prevent_sleep():
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    except Exception:
        pass


def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="True AML Dynamic Architecture Training")
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--K', type=int, default=32)
    parser.add_argument('--max-dim', type=int, default=64)
    parser.add_argument('--max-layers', type=int, default=18)
    parser.add_argument('--p-allowance', type=float, default=1.5e6)
    parser.add_argument('--out', type=str, default='aml_core_best.pth')
    return parser.parse_args()

def main():
    args = get_args()
    set_seed(args.seed)
    _prevent_sleep()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = ConvAML_CIFAR_TrueDynamic(K=args.K, max_dim=args.max_dim, max_layers=args.max_layers).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    print(f"========== TRAINING {args.max_layers}-LAYER TRUE AML DYNAMIC ==========")
    meta_loss_fn = MetaLoss(
        lambda_eff=0.0, 
        lambda_p=0.0, 
        p_allowance=args.p_allowance
    ).to(device)
    
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss, total_eff = 0, 0
        t0 = time.time()
        
        # WARMUP + EFFICIENCY/REGULARIZATION SCHEDULE
        if epoch < 5:
            current_lambda_eff = 0.0
            tau = 1.0
            w_gate = 0.0
            w_bm = 0.0
            w_dim = 0.0
        elif epoch < 10:
            current_lambda_eff = 0.0
            tau = 0.8
            w_gate = 0.02
            w_bm = 0.02
            w_dim = 0.01
        elif epoch < 15:
            current_lambda_eff = 0.01
            tau = 0.5
            w_gate = 0.05
            w_bm = 0.05
            w_dim = 0.05
        else:
            current_lambda_eff = 0.05
            tau = 0.2
            w_gate = 0.1
            w_bm = 0.1
            w_dim = 0.1
            
        meta_loss_fn.lambda_eff = current_lambda_eff
        
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs, struct, pi, diag = model(inputs, tau=tau)
            gen_model = GeneratedModel(structure=struct, weights={}, pi_logits=pi)
            
            loss_dict = meta_loss_fn(
                logits_train=outputs, targets_train=targets,
                logits_val=None, targets_val=None,
                model=gen_model, epoch_progress=epoch/args.epochs
            )
            
            loss = loss_dict["loss"]
            if w_gate > 0.0 or w_bm > 0.0 or w_dim > 0.0:
                layer_soft = diag["layer_soft"].clamp(1e-6, 1.0 - 1e-6)
                ent_gate = -(layer_soft * torch.log(layer_soft) + (1.0 - layer_soft) * torch.log(1.0 - layer_soft)).mean()
                p_bm = F.softmax(diag["bm_logits"], dim=-1).clamp(1e-6, 1.0)
                ent_bm = -(p_bm * torch.log(p_bm)).sum(dim=-1).mean()
                dim_floor = F.relu(0.4 * model.max_dim - diag["d_lengths"].mean()).mean()
                reg = -w_gate * ent_gate - w_bm * ent_bm + w_dim * dim_floor
                loss = loss + reg
            loss.backward()
            optimizer.step()
            
            total_loss += loss_dict["task_loss"].item() * inputs.size(0)
            total_eff += loss_dict.get("efficiency_cost", torch.tensor(0.0)).item() * inputs.size(0)
            
        scheduler.step()
        train_loss = total_loss / len(trainset)
        
        model.eval()
        correct, eff_params_sum = 0, 0
        bm_sums = torch.zeros(model.max_layers, 3, device=device) # [max_layers, 3]
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs, struct, _, _ = model(inputs, tau=0.1)
                correct += (outputs.argmax(-1) == targets).sum().item()
                eff_params_sum += struct.effective_params_formula().sum().item()
                bm_sums += struct.block_mode.sum(dim=0)
                
        test_acc = correct / len(testset)
        avg_eff_params = eff_params_sum / len(testset)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.out)
            print(f"  [*] Saved new best model to {args.out} (Acc: {test_acc*100:.2f}%)")
        
        print(f"Epoch {epoch+1:2d}/{args.epochs} | {time.time()-t0:5.1f}s | Task Loss: {train_loss:.4f} | Test Acc: {test_acc*100:.2f}% | Avg Used Params: {avg_eff_params/1e6:.2f}M | Tau: {tau:.1f} | Lambda: {current_lambda_eff:.4f} | Reg(g/b/d): {w_gate:.2f}/{w_bm:.2f}/{w_dim:.3f}")
        
        bm_avg = bm_sums / len(testset)
        print("  -> Block Mode Distribution (Plain / ResNet / ViT) per Layer:")
        for i in range(model.max_layers):
            p, r, v = bm_avg[i].tolist()
            print(f"     Layer {i+1}: {p*100:5.1f}% | {r*100:5.1f}% | {v*100:5.1f}%")

if __name__ == '__main__':
    print("WARNING: Eager path currently only supports Plain/ResNet block mode. ViT block is ignored in fallback.")
    main()
