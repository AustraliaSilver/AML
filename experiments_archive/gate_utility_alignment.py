import math
import os
import time
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_wikitext103_scaled import ScaledAMLCharTransformer

def run_gate_utility_alignment_study():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== GATE-UTILITY ALIGNMENT EXPERIMENT ({device}) ===")
    
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
    marginal_utilities = [] # U_8(x) = BPC_7(x) - BPC_8(x)
    
    print(f"\nEvaluating {num_samples:,} samples on Unseen test.txt for Gate-Utility Alignment:")
    
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
            
            # Forward layers 1..7
            h_curr = h
            for i in range(7):
                d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
                h_layer = model.layers[i](h_curr, bm_probs[:, i, :], d_mask_i, is_first=(i==0))
                gate_i = layer_gates[:, i:i+1, None]
                h_curr = gate_i * h_layer + (1.0 - gate_i) * h_curr
                
            # Path A: Force-7 Output
            logits_7 = model.lm_head(model.norm_final(h_curr))
            l7 = criterion(logits_7.view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # Path B: Force-8 Output
            d_mask_7 = torch.sigmoid(10.0 * (d_lengths[:, 7:8, :] - arange))
            h_layer8 = model.layers[7](h_curr, bm_probs[:, 7, :], d_mask_7, is_first=False)
            logits_8 = model.lm_head(model.norm_final(h_layer8))
            l8 = criterion(logits_8.view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # Exact continuous gate G8(x)
            g8_val = layer_gates[:, 7].cpu().numpy()
            
            # Marginal utility U_8(x) = BPC_7 - BPC_8
            bpc_7 = l7 / math.log(2)
            bpc_8 = l8 / math.log(2)
            u8 = bpc_7 - bpc_8
            
            g8_scores.extend(g8_val)
            marginal_utilities.extend(u8)
            
    g8_scores = np.array(g8_scores)
    marginal_utilities = np.array(marginal_utilities)
    
    # 1. Pearson and Spearman Correlation
    pearson_r, pearson_p = stats.pearsonr(g8_scores, marginal_utilities)
    spearman_r, spearman_p = stats.spearmanr(g8_scores, marginal_utilities)
    
    print("\n==================== 1. GATE-UTILITY CORRELATION METRICS ====================")
    print(f"Evaluated {len(g8_scores):,} independent test sequences:")
    print(f"  * Pearson Correlation (Linear)  : rho_P = {pearson_r:+.4f} (p = {pearson_p:.4e})")
    print(f"  * Spearman Correlation (Rank)   : rho_S = {spearman_r:+.4f} (p = {spearman_p:.4e})")
    
    # 2. 10 Decile Bins across G8(x) Score
    print("\n==================== 2. BINNED MARGINAL UTILITY TRAJECTORY (E[U_8 | G_8]) ====================")
    n_bins = 10
    # Bin by G8 score range: [min, max]
    bin_edges = np.linspace(g8_scores.min(), g8_scores.max(), n_bins + 1)
    
    print(f"{'Bin':<6} {'Gate Score Range (G_8)':<26} {'Sample Count':<14} {'Mean G_8':<12} {'Marginal Utility E[U_8]':<26} {'Visual Utility Bar'}")
    print("-" * 105)
    
    binned_data = []
    for b in range(n_bins):
        low, high = bin_edges[b], bin_edges[b+1]
        if b == n_bins - 1:
            mask = (g8_scores >= low) & (g8_scores <= high)
        else:
            mask = (g8_scores >= low) & (g8_scores < high)
            
        n_in_bin = np.sum(mask)
        if n_in_bin > 0:
            mean_g8 = np.mean(g8_scores[mask])
            mean_u8 = np.mean(marginal_utilities[mask])
            std_u8 = np.std(marginal_utilities[mask]) / math.sqrt(n_in_bin)
            binned_data.append((mean_g8, mean_u8, n_in_bin))
            
            # Scaled visual bar
            bar_len = int(max(0, mean_u8) * 300)
            bar = "=" * bar_len
            print(f"B{b+1:<4d} [{low:.3f} - {high:.3f}]              {n_in_bin:<14d} {mean_g8:<12.3f} {mean_u8:+8.4f} +/- {std_u8:.4f} BPC     [{bar:<30}]")
        else:
            print(f"B{b+1:<4d} [{low:.3f} - {high:.3f}]              0              ---          ---")

    # 3. High Gate vs Low Gate Contrast
    q33 = np.percentile(g8_scores, 33.3)
    q66 = np.percentile(g8_scores, 66.6)
    
    low_gate_u8 = np.mean(marginal_utilities[g8_scores <= q33])
    high_gate_u8 = np.mean(marginal_utilities[g8_scores >= q66])
    
    print("\n==================== 3. CONTRAST & EFFECT SUMMARY ====================")
    print(f"  * Low Router Confidence  (Bottom 33.3% G_8) -> Marginal Benefit of Layer 8: {low_gate_u8:+.4f} BPC")
    print(f"  * High Router Confidence (Top 33.3% G_8)    -> Marginal Benefit of Layer 8: {high_gate_u8:+.4f} BPC")
    print(f"  * Contrast Ratio                            : {high_gate_u8 / max(low_gate_u8, 1e-5):.2f}x higher benefit for high-gate samples!")
    
    print("\n[SUCCESS] Gate-Utility Alignment study completed!")

if __name__ == '__main__':
    run_gate_utility_alignment_study()
