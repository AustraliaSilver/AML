import math
import os
import time
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_wikitext103_scaled import ScaledAMLCharTransformer

def run_budget_matched_pareto_and_calibration():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== BUDGET-MATCHED PARETO CURVE & REFINED CALIBRATION ANALYSIS ({device}) ===")
    
    test_path = os.path.join('data', 'wikitext103', 'test.txt')
    with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
        test_text = f.read()
    test_tensor = torch.tensor([ord(c) % 256 for c in test_text], dtype=torch.long)
    
    # Load model
    model = ScaledAMLCharTransformer(vocab_size=256, max_layers=8, max_dim=192, seq_len=256, K=16, latent_dim=128).to(device)
    model.load_state_dict(torch.load('aml_scaled_wikitext103_best.pth', map_location=device))
    model.eval()
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    num_samples = 4000
    seq_len = 256
    batch_size = 64
    n_batches = num_samples // batch_size
    
    g8_scores = []
    losses_7 = []
    losses_8 = []
    
    # Save intermediate representations to construct exact Soft/Hard sweeps at arbitrary thresholds tau
    h_curr_list = []
    h_layer8_list = []
    y_b_list = []
    
    print(f"\nCollecting intermediate test representations across {num_samples:,} samples...")
    with torch.no_grad():
        for b in range(n_batches):
            idx = torch.randint(0, len(test_tensor) - seq_len - 1, (batch_size,))
            x_b = torch.stack([test_tensor[j:j+seq_len] for j in idx]).to(device)
            y_b = torch.stack([test_tensor[j+1:j+seq_len+1] for j in idx]).to(device)
            
            B, T = x_b.shape
            h = model.drop(model.tok_embed(x_b) + model.pos_embed[:, :T, :])
            z = model.ctx_encoder(h.transpose(1, 2))
            delta = model.structure_delta(z)
            layer_gates = torch.sigmoid(delta[:, :model.max_layers] + 2.0)
            d_lengths = 64.0 + F.softplus(torch.sigmoid(delta[:, model.max_layers:].view(B, model.max_layers, model.max_dim).mean(-1, keepdim=True) + 2.0) * model.max_dim - 64.0)
            arange = torch.arange(model.max_dim, device=device).float().view(1, 1, model.max_dim)
            bm_probs = F.gumbel_softmax(model.block_mode_head(z).view(B, model.max_layers, 4), tau=0.3, hard=True, dim=-1)
            
            # Layers 1..7
            h_curr = h
            for i in range(7):
                d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
                h_layer = model.layers[i](h_curr, bm_probs[:, i, :], d_mask_i, is_first=(i==0))
                gate_i = layer_gates[:, i:i+1, None]
                h_curr = gate_i * h_layer + (1.0 - gate_i) * h_curr
                
            # Layer 8
            d_mask_7 = torch.sigmoid(10.0 * (d_lengths[:, 7:8, :] - arange))
            h_layer8 = model.layers[7](h_curr, bm_probs[:, 7, :], d_mask_7, is_first=False)
            
            # Pure Layer 7 & Layer 8 outputs
            l7 = criterion(model.lm_head(model.norm_final(h_curr)).view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            l8 = criterion(model.lm_head(model.norm_final(h_layer8)).view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            g8_scores.extend(layer_gates[:, 7].cpu().numpy())
            losses_7.extend(l7)
            losses_8.extend(l8)
            
            h_curr_list.append(h_curr.cpu())
            h_layer8_list.append(h_layer8.cpu())
            y_b_list.append(y_b.cpu())
            
    g8_scores = np.array(g8_scores)
    bpc_7 = np.array(losses_7) / math.log(2)
    bpc_8 = np.array(losses_8) / math.log(2)
    u8 = bpc_7 - bpc_8
    
    # ==================== 1. BUDGET-MATCHED PARETO FRONTIER SWEEP ====================
    print("\n==================== 1. FULL BUDGET-MATCHED PARETO SWEEP ====================")
    print("Sweeping allocation budgets from 87.5% (Static-7) to 100.0% (Static-8) across 4 Policy Classes:")
    print("1. Random Skipping | 2. Hard-Threshold AML | 3. Scaled Soft AML | 4. Hard-Routing Oracle")
    
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    budgets_to_match = [88.0, 90.0, 92.7, 95.0, 97.5, 100.0]
    
    # Evaluate Hard AML across thresholds tau in [0.1 .. 0.9]
    print("\n[A] Discrete Hard-AML Sweep over Thresholds (tau in [0.1, 0.9]):")
    print(f"{'Threshold (tau)':<18} {'P(L=8 Allocation)':<22} {'Rel. FLOPs':<15} {'Test BPC':<15}")
    print("-" * 70)
    hard_sweep_results = []
    
    for tau in thresholds:
        # Decision: L=8 if G8 >= tau
        dec = (g8_scores >= tau).astype(int)
        alloc_rate = np.mean(dec) * 100
        rel_flops = (7.0 + np.mean(dec)) / 8.0 * 100
        
        # BPC per sample
        bpc_samples = np.where(dec == 1, bpc_8, bpc_7)
        mean_bpc = np.mean(bpc_samples)
        hard_sweep_results.append((rel_flops, mean_bpc))
        print(f"tau = {tau:4.2f}            {alloc_rate:5.2f}%                 {rel_flops:5.2f}%         {mean_bpc:.4f} BPC")

    # Evaluate Scaled Soft AML across multiplier alpha in [0.2 .. 1.4]
    print("\n[B] Continuous Soft-AML Sweep over Soft Scaling Multipliers (alpha in [0.2, 1.4]):")
    print(f"{'Multiplier (alpha)':<20} {'Effective FLOPs':<18} {'Test BPC':<15}")
    print("-" * 55)
    soft_sweep_results = []
    
    for alpha in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]:
        # G8_scaled = clamp(alpha * G8, 0, 1)
        g8_scaled = np.clip(alpha * g8_scores, 0.0, 1.0)
        rel_flops = (7.0 + np.mean(g8_scaled)) / 8.0 * 100
        
        # Calculate BPC by soft blending predictions
        bpc_soft_list = []
        with torch.no_grad():
            for b in range(n_batches):
                h_c = h_curr_list[b].to(device)
                h_8 = h_layer8_list[b].to(device)
                y_ = y_b_list[b].to(device)
                g_ = torch.tensor(g8_scaled[b*batch_size:(b+1)*batch_size], device=device, dtype=torch.float).view(-1, 1, 1)
                
                h_blend = g_ * h_8 + (1.0 - g_) * h_c
                l_s = criterion(model.lm_head(model.norm_final(h_blend)).view(-1, 256), y_.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
                bpc_soft_list.extend(l_s / math.log(2))
                
        mean_bpc_soft = np.mean(bpc_soft_list)
        soft_sweep_results.append((rel_flops, mean_bpc_soft))
        print(f"alpha = {alpha:4.2f}               {rel_flops:5.2f}%             {mean_bpc_soft:.4f} BPC")

    # ==================== 2. EXACT BUDGET-MATCHED DIRECT COMPARISON ====================
    print("\n==================== 2. EXACT BUDGET-MATCHED HEAD-TO-HEAD COMPARISON ====================")
    # Fix exact budget = 92.7% (Average depth L = 7.416)
    target_budget = 92.7
    
    # 1. Random Matching 92.7%
    rand_alloc = (target_budget - 87.5) / 12.5 # fraction of L=8
    rand_bpc_matched = (1.0 - rand_alloc) * np.mean(bpc_7) + rand_alloc * np.mean(bpc_8)
    
    # 2. Hard AML Matching 92.7% (tau where P(L=8) = rand_alloc)
    tau_matched = np.percentile(g8_scores, (1.0 - rand_alloc) * 100)
    hard_dec_matched = (g8_scores >= tau_matched).astype(int)
    hard_bpc_matched = np.mean(np.where(hard_dec_matched == 1, bpc_8, bpc_7))
    
    # 3. Soft AML Matching 92.7% (alpha where mean(G8_scaled) = rand_alloc)
    alpha_matched = rand_alloc / np.mean(g8_scores)
    g8_soft_matched = np.clip(alpha_matched * g8_scores, 0.0, 1.0)
    bpc_soft_matched_list = []
    with torch.no_grad():
        for b in range(n_batches):
            h_c = h_curr_list[b].to(device)
            h_8 = h_layer8_list[b].to(device)
            y_ = y_b_list[b].to(device)
            g_ = torch.tensor(g8_soft_matched[b*batch_size:(b+1)*batch_size], device=device, dtype=torch.float).view(-1, 1, 1)
            h_blend = g_ * h_8 + (1.0 - g_) * h_c
            l_s = criterion(model.lm_head(model.norm_final(h_blend)).view(-1, 256), y_.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            bpc_soft_matched_list.extend(l_s / math.log(2))
    soft_bpc_matched = np.mean(bpc_soft_matched_list)
    
    # 4. Hard-Routing Oracle Matching 92.7%
    u8_threshold = np.percentile(u8, (1.0 - rand_alloc) * 100)
    oracle_dec_matched = (u8 >= u8_threshold).astype(int)
    oracle_bpc_matched = np.mean(np.where(oracle_dec_matched == 1, bpc_8, bpc_7))

    print(f"Policy Class (All Exactly Matched at {target_budget:.1f}% FLOPs)    Test BPC    Delta vs Random")
    print("-" * 75)
    print(f"1. Random Skipping Policy                       {rand_bpc_matched:.4f} BPC    0.0000 BPC (Baseline)")
    print(f"2. Discrete Hard-AML Router                     {hard_bpc_matched:.4f} BPC   {hard_bpc_matched - rand_bpc_matched:+.4f} BPC")
    print(f"3. Hard-Routing Oracle (Policy Upper Bound)     {oracle_bpc_matched:.4f} BPC   {oracle_bpc_matched - rand_bpc_matched:+.4f} BPC")
    print(f"4. Continuous Soft-AML Router                   {soft_bpc_matched:.4f} BPC   {soft_bpc_matched - rand_bpc_matched:+.4f} BPC")
    
    print(f"\nExact Head-to-Head Insight:")
    print(f"  * At strictly equal 92.7% FLOPs, Hard-AML outperforms Random by {rand_bpc_matched - hard_bpc_matched:.4f} BPC.")
    print(f"  * Soft-AML captures {((rand_bpc_matched - soft_bpc_matched)/(rand_bpc_matched - oracle_bpc_matched))*100:.1f}% of the theoretical Oracle advantage.")

    # ==================== 3. REFINED CALIBRATION CURVE WITH BOOTSTRAP CIs ====================
    print("\n==================== 3. EMPIRICAL CALIBRATION CURVE WITH BOOTSTRAP CIs ====================")
    n_bins = 8
    # Equal-frequency binning (Quantile bins so all bins have robust N ~ 500 samples!)
    quantiles = np.linspace(0, 100, n_bins + 1)
    q_edges = np.percentile(g8_scores, quantiles)
    
    print(f"{'Bin':<6} {'Gate Quantile Range':<24} {'Sample Count':<14} {'Mean G_8':<12} {'Marginal Utility E[U_8]':<26} {'95% Bootstrap CI'}")
    print("-" * 105)
    
    n_boot = 1000
    for b in range(n_bins):
        low, high = q_edges[b], q_edges[b+1]
        if b == n_bins - 1:
            mask = (g8_scores >= low) & (g8_scores <= high)
        else:
            mask = (g8_scores >= low) & (g8_scores < high)
            
        n_cnt = np.sum(mask)
        mean_g = np.mean(g8_scores[mask])
        u_vals = u8[mask]
        mean_u = np.mean(u_vals)
        
        # Bootstrap CI for this bin
        boot_means = [np.mean(np.random.choice(u_vals, size=n_cnt, replace=True)) for _ in range(n_boot)]
        ci_l, ci_h = np.percentile(boot_means, [2.5, 97.5])
        
        print(f"Q{b+1:<4d} [{low:.3f} - {high:.3f}]              {n_cnt:<14d} {mean_g:<12.3f} {mean_u:+8.4f} BPC             [{ci_l:+7.4f}, {ci_h:+7.4f}]")

    print("\n[SUCCESS] Budget-Matched Pareto Sweep and Refined Calibration Analysis completed!")

if __name__ == '__main__':
    run_budget_matched_pareto_and_calibration()
