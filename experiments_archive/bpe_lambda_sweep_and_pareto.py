import os
import math
import time
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from aml_tokenizer import AMLBPETokenizer
from train_bpe_aml import AMLBPETransformer, set_seed

def run_bpe_lambda_sweep_and_pareto():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== BPE LAMBDA-SWEEP & PARETO FRONTIER BENCHMARK ({device}) ===")
    
    # 1. Load exact tokenized data & compression ratio
    tokenizer = AMLBPETokenizer().load("aml_bpe_tokenizer.json")
    vocab_size = tokenizer.vocab_size
    
    with open("data/wikitext103/test.txt", "r", encoding="utf-8", errors="ignore") as f:
        test_raw_chars = len(f.read())
        
    train_tokens = torch.load("data/wikitext103_tokenized/train_bpe.pt", weights_only=True)
    valid_tokens = torch.load("data/wikitext103_tokenized/valid_bpe.pt", weights_only=True)
    test_tokens  = torch.load("data/wikitext103_tokenized/test_bpe.pt", weights_only=True)
    
    r_exact = test_raw_chars / len(test_tokens) # 3.9233 chars/token
    print(f"Loaded Datasets: Train={len(train_tokens):,} | Valid={len(valid_tokens):,} | Test={len(test_tokens):,} tokens (r={r_exact:.4f})")
    
    # Grid of lambda penalties: 0.0 (Unconstrained), 0.003 (Mild), 0.01 (Moderate), 0.03 (Strong), 0.08 (Very strong)
    lambda_grid = [0.000, 0.003, 0.010, 0.030, 0.080]
    target_rel_flops = 0.927 # 92.7% target FLOPs
    
    sweep_results = {}
    
    batch_size = 48
    seq_len = 256
    steps_per_epoch = 300
    epochs = 6
    
    for l_idx, lambda_b in enumerate(lambda_grid):
        print(f"\n==================== TRAINING BPE AML WITH LAMBDA = {lambda_b:.4f} ====================")
        set_seed(2026 + l_idx)
        model = AMLBPETransformer(vocab_size=vocab_size, max_layers=8, max_dim=256, seq_len=256, K=16, latent_dim=128).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=8e-5)
        criterion = nn.CrossEntropyLoss()
        
        ckpt_name = f"aml_bpe_sweep_lambda_{lambda_b:.4f}.pth"
        best_val_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            model.train()
            train_ce_acc = 0.0
            g8_acc = 0.0
            for step in range(steps_per_epoch):
                idx = torch.randint(0, len(train_tokens) - seq_len - 1, (batch_size,))
                x = torch.stack([train_tokens[j:j+seq_len] for j in idx]).to(device)
                y = torch.stack([train_tokens[j+1:j+seq_len+1] for j in idx]).to(device)
                
                optimizer.zero_grad()
                logits, diag = model(x, tau=0.5)
                ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                
                # Expected FLOPs penalty
                g8_batch = diag['layer_gates'][:, 7]
                expected_flops = (7.0 + g8_batch.mean()) / 8.0
                budget_loss = lambda_b * ((expected_flops / target_rel_flops - 1.0) ** 2)
                
                loss = ce_loss + budget_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                train_ce_acc += ce_loss.item()
                g8_acc += g8_batch.mean().item()
                
            scheduler.step()
            
            # Validation on valid_tokens
            model.eval()
            val_losses = []
            val_g8s = []
            with torch.no_grad():
                for v_step in range(30):
                    v_idx = torch.randint(0, len(valid_tokens) - seq_len - 1, (batch_size,))
                    vx = torch.stack([valid_tokens[j:j+seq_len] for j in v_idx]).to(device)
                    vy = torch.stack([valid_tokens[j+1:j+seq_len+1] for j in v_idx]).to(device)
                    v_logits, v_diag = model(vx, tau=0.3)
                    val_losses.append(criterion(v_logits.view(-1, vocab_size), vy.view(-1)).item())
                    val_g8s.append(v_diag['layer_gates'][:, 7].mean().item())
                    
            val_loss = float(np.mean(val_losses))
            mean_g8_val = float(np.mean(val_g8s))
            rel_flops_val = (7.0 + mean_g8_val) / 8.0 * 100
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), ckpt_name)
                
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_ce_acc/steps_per_epoch:.4f} | Valid Loss: {val_loss:.4f} (PPL {math.exp(min(val_loss, 20)):5.1f}) | Mean G8: {mean_g8_val*100:5.2f}% | FLOPs: {rel_flops_val:5.2f}%")

        # --- EVALUATION ON UNSEEN TEST_TOKENS ---
        model.load_state_dict(torch.load(ckpt_name, map_location=device, weights_only=True))
        model.eval()
        
        num_eval = 2400
        n_eval_batches = num_eval // batch_size
        crit_elem = nn.CrossEntropyLoss(reduction='none')
        
        g8_eval = []
        l7_eval = []
        l8_eval = []
        l_soft_eval = []
        l_hard_eval = []
        
        h_curr_list = []
        h_layer8_list = []
        y_b_list = []
        
        with torch.no_grad():
            for b in range(n_eval_batches):
                idx = torch.randint(0, len(test_tokens) - seq_len - 1, (batch_size,))
                x_b = torch.stack([test_tokens[j:j+seq_len] for j in idx]).to(device)
                y_b = torch.stack([test_tokens[j+1:j+seq_len+1] for j in idx]).to(device)
                
                B, T = x_b.shape
                h = model.drop(model.tok_embed(x_b) + model.pos_embed[:, :T, :])
                z = model.ctx_encoder(h[:, -1, :])
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
                    
                l7 = crit_elem(model.lm_head(model.norm_final(h_curr)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                
                d_mask_7 = torch.sigmoid(10.0 * (d_lengths[:, 7:8, :] - arange))
                h_layer8 = model.layers[7](h_curr, bm_probs[:, 7, :], d_mask_7, is_first=False)
                l8 = crit_elem(model.lm_head(model.norm_final(h_layer8)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                
                g8 = layer_gates[:, 7:8, None]
                h_soft = g8 * h_layer8 + (1.0 - g8) * h_curr
                l_soft = crit_elem(model.lm_head(model.norm_final(h_soft)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                
                hard_g8 = (g8 >= 0.5).float()
                h_hard = hard_g8 * h_layer8 + (1.0 - hard_g8) * h_curr
                l_hard = crit_elem(model.lm_head(model.norm_final(h_hard)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                
                g8_eval.extend(layer_gates[:, 7].cpu().numpy())
                l7_eval.extend(l7)
                l8_eval.extend(l8)
                l_soft_eval.extend(l_soft)
                l_hard_eval.extend(l_hard)
                h_curr_list.append(h_curr.cpu())
                h_layer8_list.append(h_layer8.cpu())
                y_b_list.append(y_b.cpu())

        g8_eval = np.array(g8_eval)
        l7_arr = np.array(l7_eval)
        l8_arr = np.array(l8_eval)
        l_soft_arr = np.array(l_soft_eval)
        l_hard_arr = np.array(l_hard_eval)
        u8 = l7_arr - l8_arr # nats
        y_true_binary = (u8 > 0).astype(int)
        
        mean_g8 = np.mean(g8_eval)
        std_g8 = np.std(g8_eval)
        p_alloc = np.mean(g8_eval >= 0.5)
        rel_flops_hard = (7.0 + p_alloc) / 8.0 * 100
        rel_flops_soft = (7.0 + mean_g8) / 8.0 * 100
        
        # Spearman & Pearson CUA
        rho_s, _ = stats.spearmanr(g8_eval, u8) if std_g8 > 1e-4 else (0.0, 1.0)
        rho_p, _ = stats.pearsonr(g8_eval, u8) if std_g8 > 1e-4 else (0.0, 1.0)
        
        # AUROC / AUPRC
        auroc = roc_auc_score(y_true_binary, g8_eval) if (len(np.unique(y_true_binary)) > 1 and std_g8 > 1e-4) else 0.5
        auprc = average_precision_score(y_true_binary, g8_eval) if (len(np.unique(y_true_binary)) > 1 and std_g8 > 1e-4) else 0.5
        
        # Causal Intervention on Subgroup L=8 and L=7
        mask_nat8 = (g8_eval >= 0.5)
        mask_nat7 = (g8_eval < 0.5)
        delta_deprived_bpc = (np.mean(l7_arr[mask_nat8]) - np.mean(l8_arr[mask_nat8])) / math.log(2) / r_exact if np.sum(mask_nat8) > 0 else 0.0
        delta_wasted_bpc = (np.mean(l8_arr[mask_nat7]) - np.mean(l7_arr[mask_nat7])) / math.log(2) / r_exact if np.sum(mask_nat7) > 0 else 0.0
        
        # Permutation Test
        g8_perm = np.random.permutation(g8_eval)
        bpc_perm_list = []
        with torch.no_grad():
            for b in range(n_eval_batches):
                h_c = h_curr_list[b].to(device)
                h_8 = h_layer8_list[b].to(device)
                y_ = y_b_list[b].to(device)
                g_ = torch.tensor(g8_perm[b*batch_size:(b+1)*batch_size], device=device, dtype=torch.float).view(-1, 1, 1)
                h_blend = g_ * h_8 + (1.0 - g_) * h_c
                l_s = crit_elem(model.lm_head(model.norm_final(h_blend)).view(-1, vocab_size), y_.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                bpc_perm_list.extend(l_s / math.log(2) / r_exact)
        delta_perm = np.mean(bpc_perm_list) - (np.mean(l_soft_arr) / math.log(2) / r_exact)
        
        sweep_results[lambda_b] = {
            'loss': np.mean(l_soft_arr),
            'ppl': math.exp(np.mean(l_soft_arr)),
            'ebpc': np.mean(l_soft_arr) / math.log(2) / r_exact,
            'mean_g8': mean_g8,
            'std_g8': std_g8,
            'p_alloc': p_alloc,
            'flops_soft': rel_flops_soft,
            'flops_hard': rel_flops_hard,
            'cua_s': rho_s,
            'cua_p': rho_p,
            'auroc': auroc,
            'auprc': auprc,
            'delta_deprived_bpc': delta_deprived_bpc,
            'delta_wasted_bpc': delta_wasted_bpc,
            'delta_perm': delta_perm
        }

    # ==================== FINAL SYNTHESIS TABLE FOR PAPER ====================
    print("\n" + "="*115)
    print("FINAL BPE LAMBDA-SWEEP & PARETO FRONTIER BENCHMARK SUMMARY (test.txt, r = 3.9233 chars/token)")
    print("="*115)
    print(f"{'Lambda':<8} {'Mean G8':<11} {'Std(G8)':<10} {'% L=8':<9} {'FLOPs':<9} {'Test PPL':<11} {'eBPC':<12} {'CUA (rho_S)':<14} {'AUROC':<9} {'Delta_Perm'}")
    print("-" * 115)
    for l_b, r in sweep_results.items():
        print(f"{l_b:<8.3f} {r['mean_g8']*100:<10.2f}% {r['std_g8']:<10.4f} {r['p_alloc']*100:<8.1f}% {r['flops_soft']:<8.2f}% {r['ppl']:<11.2f} {r['ebpc']:<11.4f}  {r['cua_s']:+12.4f}  {r['auroc']:<9.4f} {r['delta_perm']:+8.4f}")

if __name__ == '__main__':
    run_bpe_lambda_sweep_and_pareto()
