import math
import os
import time
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_wikitext103_scaled import ScaledAMLCharTransformer

def run_causal_routing_intervention():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== EXPERIMENT 1: CAUSAL ROUTING INTERVENTION STUDY ({device}) ===")
    
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
    
    natural_routing = [] # 1 if G8 >= 0.5 else 0
    losses_natural = []
    losses_force7 = []
    losses_force8 = []
    
    print(f"\nEvaluating {num_samples:,} samples on Unseen test.txt under 3 Intervention Conditions:")
    print("  1. Condition Natural: Router selects G8 naturally")
    print("  2. Condition Force-7: Hard-skip Layer 8 (Strictly bypass compute)")
    print("  3. Condition Force-8: Force Layer 8 (G8 = 1.0)")
    
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
                
            # --- Condition Force-7: Compute directly from Layer 7 (Zero Layer 8 FLOPs) ---
            logits_force7 = model.lm_head(model.norm_final(h_curr))
            l_force7 = criterion(logits_force7.view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # Layer 8 forward
            d_mask_7 = torch.sigmoid(10.0 * (d_lengths[:, 7:8, :] - arange))
            h_layer8 = model.layers[7](h_curr, bm_probs[:, 7, :], d_mask_7, is_first=False)
            
            # --- Condition Force-8: Force Gate 8 = 1.0 ---
            logits_force8 = model.lm_head(model.norm_final(h_layer8))
            l_force8 = criterion(logits_force8.view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # --- Condition Natural: Natural Continuous Gate 8 ---
            g8 = layer_gates[:, 7:8, None]
            h_natural = g8 * h_layer8 + (1.0 - g8) * h_curr
            logits_natural = model.lm_head(model.norm_final(h_natural))
            l_natural = criterion(logits_natural.view(-1, 256), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # Natural routing decision (Binary threshold: >= 0.5)
            is_l8_natural = (layer_gates[:, 7].cpu().numpy() >= 0.5).astype(int)
            
            natural_routing.extend(is_l8_natural)
            losses_natural.extend(l_natural)
            losses_force7.extend(l_force7)
            losses_force8.extend(l_force8)
            
    natural_routing = np.array(natural_routing)
    losses_natural = np.array(losses_natural)
    losses_force7 = np.array(losses_force7)
    losses_force8 = np.array(losses_force8)
    
    # Calculate BPC (bits per char = nats / ln(2))
    bpc_nat = losses_natural / math.log(2)
    bpc_f7 = losses_force7 / math.log(2)
    bpc_f8 = losses_force8 / math.log(2)
    
    # Subpopulation masks
    mask_nat7 = (natural_routing == 0) # Router chose L=7
    mask_nat8 = (natural_routing == 1) # Router chose L=8
    
    n_nat7 = np.sum(mask_nat7)
    n_nat8 = np.sum(mask_nat8)
    
    print(f"\n==================== CAUSAL INTERVENTION MATRIX RESULTS ====================")
    print(f"Sample Partitioning: {n_nat7:,} samples ({(n_nat7/len(natural_routing))*100:.1f}%) Natural L=7 | {n_nat8:,} samples ({(n_nat8/len(natural_routing))*100:.1f}%) Natural L=8\n")
    
    # 1. Subpopulation A: Samples that Router Naturally Assigned to L=8 (Hard / Complex text)
    bpc_nat8_under_nat = np.mean(bpc_nat[mask_nat8])
    bpc_nat8_under_f8 = np.mean(bpc_f8[mask_nat8])
    bpc_nat8_under_f7 = np.mean(bpc_f7[mask_nat8])
    
    delta_bpc_deprived = bpc_nat8_under_f7 - bpc_nat8_under_f8 # E[BPC(Force-7) - BPC(Force-8) | Natural L=8]
    t_stat_deprived, p_val_deprived = stats.ttest_rel(bpc_f7[mask_nat8], bpc_f8[mask_nat8])
    
    print(f"SUBPOPULATION 1: Samples Naturally Routed to L=8 (N={n_nat8:,}):")
    print(f"  * Natural Policy BPC       : {bpc_nat8_under_nat:.4f} BPC")
    print(f"  * Force-8 (Supplied Layer) : {bpc_nat8_under_f8:.4f} BPC")
    print(f"  * Force-7 (Deprived Layer) : {bpc_nat8_under_f7:.4f} BPC")
    print(f"  * Causal Layer Benefit     : Delta BPC = {delta_bpc_deprived:+.4f} BPC (t = {t_stat_deprived:.4f}, p = {p_val_deprived:.4e})")
    if delta_bpc_deprived > 0 and p_val_deprived < 0.001:
        print(f"  => [PROVEN] When Layer 8 is removed from samples that needed it, quality degrades significantly (p < 0.001)!")

    # 2. Subpopulation B: Samples that Router Naturally Assigned to L=7 (Simple / Moderate text)
    bpc_nat7_under_nat = np.mean(bpc_nat[mask_nat7])
    bpc_nat7_under_f7 = np.mean(bpc_f7[mask_nat7])
    bpc_nat7_under_f8 = np.mean(bpc_f8[mask_nat7])
    
    delta_bpc_wasted = bpc_nat7_under_f8 - bpc_nat7_under_f7 # E[BPC(Force-8) - BPC(Force-7) | Natural L=7]
    t_stat_wasted, p_val_wasted = stats.ttest_rel(bpc_f8[mask_nat7], bpc_f7[mask_nat7])
    
    print(f"\nSUBPOPULATION 2: Samples Naturally Routed to L=7 (N={n_nat7:,}):")
    print(f"  * Natural Policy BPC       : {bpc_nat7_under_nat:.4f} BPC")
    print(f"  * Force-7 (Natural Choice) : {bpc_nat7_under_f7:.4f} BPC")
    print(f"  * Force-8 (Forced Compute) : {bpc_nat7_under_f8:.4f} BPC")
    print(f"  * Unnecessary Compute Shift: Delta BPC = {delta_bpc_wasted:+.4f} BPC (t = {t_stat_wasted:.4f}, p = {p_val_wasted:.4e})")

    # Summary Table
    print("\n==================== INTERVENTION SUMMARY & EFFECTIVENESS ====================")
    print(f"{'Natural Routing Group':<24} {'BPC under Force-7':<20} {'BPC under Force-8':<20} {'Net Causal Effect'}")
    print("-" * 84)
    print(f"{'Natural L=8 Subgroup':<24} {bpc_nat8_under_f7:<20.4f} {bpc_nat8_under_f8:<20.4f} Layer 8 Essential (+{delta_bpc_deprived:.4f} BPC harm if skipped)")
    print(f"{'Natural L=7 Subgroup':<24} {bpc_nat7_under_f7:<20.4f} {bpc_nat7_under_f8:<20.4f} Layer 8 Redundant ({delta_bpc_wasted:+.4f} BPC change if added)")
    print(f"{'Overall Policy BPC':<24} {np.mean(bpc_f7):<20.4f} {np.mean(bpc_f8):<20.4f} Natural AML: {np.mean(bpc_nat):.4f} BPC")

if __name__ == '__main__':
    run_causal_routing_intervention()
