import math
import os
import time
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from retrain_and_verify_seed2026 import ScaledAMLCharTransformer, set_seed

# Sửa kiến trúc để loại bỏ bias cố định +2.0
class ScaledAMLCharTransformerUnbiased(ScaledAMLCharTransformer):
    def forward(self, x, tau=0.3):
        B, T = x.shape
        h = self.drop(self.tok_embed(x) + self.pos_embed[:, :T, :])
        z = self.ctx_encoder(h.transpose(1, 2))
        
        pi = F.softmax(self.proto_selector(z), dim=-1)
        delta = self.structure_delta(z)
        
        # Khởi tạo không bias (+0.0) để router có không gian học cả 2 chiều
        layer_gates = torch.sigmoid(delta[:, :self.max_layers])
        
        d_raw = torch.sigmoid(delta[:, self.max_layers:].view(B, self.max_layers, self.max_dim).mean(-1, keepdim=True))
        d_lengths = 64.0 + F.softplus(d_raw * self.max_dim - 64.0)
        arange = torch.arange(self.max_dim, device=x.device).float().view(1, 1, self.max_dim)
        
        bm_logits = self.block_mode_head(z).view(B, self.max_layers, 4)
        bm_probs = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1)
        
        for i, layer in enumerate(self.layers):
            h_prev = h
            d_mask = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
            h_layer = layer(h, bm_probs[:, i, :], d_mask, is_first=(i==0))
            gate = layer_gates[:, i:i+1, None]
            h = gate * h_layer + (1.0 - gate) * h_prev
            
        logits = self.lm_head(self.norm_final(h))
        return logits, {'pi': pi, 'layer_gates': layer_gates, 'd_lengths': d_lengths, 'bm_probs': bm_probs}

def run_budget_regularization_sweep():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== COMPREHENSIVE FLOPs BUDGET REGULARIZATION SWEEP ({device}) ===")
    
    train_path = os.path.join('data', 'wikitext103', 'train.txt')
    valid_path = os.path.join('data', 'wikitext103', 'valid.txt')
    test_path  = os.path.join('data', 'wikitext103', 'test.txt')
    
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_text = f.read(15000000)
    with open(valid_path, 'r', encoding='utf-8', errors='ignore') as f:
        valid_text = f.read()
    with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
        test_text = f.read()
        
    train_tensor = torch.tensor([ord(c) % 256 for c in train_text], dtype=torch.long)
    valid_tensor = torch.tensor([ord(c) % 256 for c in valid_text], dtype=torch.long)
    test_tensor  = torch.tensor([ord(c) % 256 for c in test_text], dtype=torch.long)
    
    # Sweep Lambda: 0.0 (Baseline), 0.01 (Small), 0.05 (Target 92.7%), 0.15 (Large)
    lambda_list = [0.00, 0.01, 0.05, 0.15]
    target_rel_flops = 0.927 # 92.7% tương đương E[G8] = (92.7% - 87.5%)/12.5% = 0.416
    target_g8 = (target_rel_flops - 0.875) / 0.125
    
    results = {}
    
    batch_size = 64
    seq_len = 256
    steps_per_epoch = 300
    epochs = 6
    
    for l_idx, lambda_b in enumerate(lambda_list):
        print(f"\n==================== TRAINING WITH LAMBDA = {lambda_b:.3f} (Target E[G8] = {target_g8:.3f}) ====================")
        set_seed(2026 + l_idx)
        model = ScaledAMLCharTransformerUnbiased(vocab_size=256, max_layers=8, max_dim=192, seq_len=256, K=16, latent_dim=128).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.CrossEntropyLoss()
        
        ckpt_name = f'aml_sweep_lambda_{lambda_b:.2f}.pth'
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_ce_acc = 0.0
            g8_acc = 0.0
            for step in range(steps_per_epoch):
                idx = torch.randint(0, len(train_tensor) - seq_len - 1, (batch_size,))
                x = torch.stack([train_tensor[j:j+seq_len] for j in idx]).to(device)
                y = torch.stack([train_tensor[j+1:j+seq_len+1] for j in idx]).to(device)
                
                optimizer.zero_grad()
                logits, diag = model(x, tau=0.5)
                ce_loss = criterion(logits.view(-1, 256), y.view(-1))
                
                # Actual FLOPs Budget Loss: (E[C(x)] / C_target - 1.0)^2
                g8_batch = diag['layer_gates'][:, 7]
                expected_flops_ratio = (7.0 + g8_batch.mean()) / 8.0
                budget_loss = lambda_b * ((expected_flops_ratio / target_rel_flops - 1.0) ** 2)
                
                loss = ce_loss + budget_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_ce_acc += ce_loss.item()
                g8_acc += g8_batch.mean().item()
                
            scheduler.step()
            
            # Validation on valid.txt ONLY
            model.eval()
            val_losses = []
            val_g8s = []
            with torch.no_grad():
                for v_step in range(30):
                    v_idx = torch.randint(0, len(valid_tensor) - seq_len - 1, (batch_size,))
                    vx = torch.stack([valid_tensor[j:j+seq_len] for j in v_idx]).to(device)
                    vy = torch.stack([valid_tensor[j+1:j+seq_len+1] for j in v_idx]).to(device)
                    v_logits, v_diag = model(vx, tau=0.3)
                    val_losses.append(criterion(v_logits.view(-1, 256), vy.view(-1)).item())
                    val_g8s.append(v_diag['layer_gates'][:, 7].mean().item())
                    
            val_bpc = np.mean(val_losses) / math.log(2)
            mean_g8_val = np.mean(val_g8s)
            
            if np.mean(val_losses) < best_val_loss:
                best_val_loss = np.mean(val_losses)
                torch.save(model.state_dict(), ckpt_name)
                
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train BPC: {train_ce_acc/steps_per_epoch/math.log(2):.4f} | Valid BPC: {val_bpc:.4f} | Mean G8: {mean_g8_val*100:5.2f}% | Rel FLOPs: {((7.0+mean_g8_val)/8.0)*100:5.2f}%")

        # --- EVALUATION ON UNSEEN TEST.TXT ---
        model.load_state_dict(torch.load(ckpt_name, map_location=device))
        model.eval()
        
        num_eval = 2000
        n_eval_batches = num_eval // batch_size
        crit_elem = nn.CrossEntropyLoss(reduction='none')
        
        g8_eval = []
        l7_eval = []
        l8_eval = []
        l_soft_eval = []
        
        with torch.no_grad():
            for b in range(n_eval_batches):
                idx = torch.randint(0, len(test_tensor) - seq_len - 1, (batch_size,))
                x_b = torch.stack([test_tensor[j:j+seq_len] for j in idx]).to(device)
                y_b = torch.stack([test_tensor[j+1:j+seq_len+1] for j in idx]).to(device)
                
                B, T = x_b.shape
                h = model.drop(model.tok_embed(x_b) + model.pos_embed[:, :T, :])
                z = model.ctx_encoder(h.transpose(1, 2))
                delta = model.structure_delta(z)
                layer_gates = torch.sigmoid(delta[:, :model.max_layers])
                d_raw = torch.sigmoid(delta[:, model.max_layers:].view(B, model.max_layers, model.max_dim).mean(-1, keepdim=True))
                d_lengths = 64.0 + F.softplus(d_raw * model.max_dim - 64.0)
                arange = torch.arange(model.max_dim, device=device).float().view(1, 1, model.max_dim)
                bm_probs = F.gumbel_softmax(model.block_mode_head(z).view(B, model.max_layers, 4), tau=0.3, hard=True, dim=-1)
                
                h_curr = h
                for i in range(7):
                    d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
                    h_layer = model.layers[i](h_curr, bm_probs[:, i, :], d_mask_i, is_first=(i==0))
                    gate_i = layer_gates[:, i:i+1, None]
                    h_curr = gate_i * h_layer + (1.0 - gate_i) * h_curr
                    
                l7 = crit_elem(model.lm_head(model.norm_final(h_curr)).view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                
                d_mask_7 = torch.sigmoid(10.0 * (d_lengths[:, 7:8, :] - arange))
                h_layer8 = model.layers[7](h_curr, bm_probs[:, 7, :], d_mask_7, is_first=False)
                l8 = crit_elem(model.lm_head(model.norm_final(h_layer8)).view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                
                g8 = layer_gates[:, 7:8, None]
                h_soft = g8 * h_layer8 + (1.0 - g8) * h_curr
                l_soft = crit_elem(model.lm_head(model.norm_final(h_soft)).view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                
                g8_eval.extend(layer_gates[:, 7].cpu().numpy())
                l7_eval.extend(l7)
                l8_eval.extend(l8)
                l_soft_eval.extend(l_soft)
                
        g8_eval = np.array(g8_eval)
        bpc_7 = np.array(l7_eval) / math.log(2)
        bpc_8 = np.array(l8_eval) / math.log(2)
        bpc_soft = np.array(l_soft_eval) / math.log(2)
        u8 = bpc_7 - bpc_8
        
        mean_g8 = np.mean(g8_eval)
        std_g8 = np.std(g8_eval)
        frac_l7 = np.mean(g8_eval < 0.5) * 100
        frac_l8 = np.mean(g8_eval >= 0.5) * 100
        rel_flops = (7.0 + mean_g8) / 8.0 * 100
        
        rho_s, _ = stats.spearmanr(g8_eval, u8) if std_g8 > 1e-4 else (0.0, 1.0)
        
        results[lambda_b] = {
            'bpc_soft': np.mean(bpc_soft),
            'mean_g8': mean_g8,
            'std_g8': std_g8,
            'frac_l7': frac_l7,
            'frac_l8': frac_l8,
            'rel_flops': rel_flops,
            'cua_spearman': rho_s
        }

    print("\n==================== FINAL BUDGET REGULARIZATION SWEEP SUMMARY ====================")
    print(f"{'Lambda':<10} {'Mean G8':<14} {'Std(G8)':<12} {'% L=7':<10} {'% L=8':<10} {'Rel FLOPs':<14} {'Test BPC':<14} {'CUA_8 (rho_S)'}")
    print("-" * 95)
    for l_b, res in results.items():
        print(f"{l_b:<10.2f} {res['mean_g8']*100:<13.2f}% {res['std_g8']:<12.4f} {res['frac_l7']:<9.1f}% {res['frac_l8']:<9.1f}% {res['rel_flops']:<13.2f}% {res['bpc_soft']:<13.4f} BPC {res['cua_spearman']:+8.4f}")

if __name__ == '__main__':
    run_budget_regularization_sweep()
