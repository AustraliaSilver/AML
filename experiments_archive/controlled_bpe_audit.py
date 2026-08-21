import os
import math
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from aml_tokenizer import AMLBPETokenizer
from train_bpe_aml import AMLBPETransformer

def run_controlled_bpe_checkpoint_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== CONTROLLED DEEP AUDIT ON BEST BPE CHECKPOINT ({device}) ===")
    
    # 1. Đo lường chính xác r_exact trên test.txt
    with open("data/wikitext103/test.txt", "r", encoding="utf-8", errors="ignore") as f:
        test_raw_chars = len(f.read())
        
    test_tokens_path = "data/wikitext103_tokenized/test_bpe.pt"
    test_tokens = torch.load(test_tokens_path, weights_only=True)
    test_total_tokens = len(test_tokens)
    
    r_exact = test_raw_chars / test_total_tokens
    print(f"Dataset Verification: {test_raw_chars:,} chars / {test_total_tokens:,} tokens -> r_exact = {r_exact:.4f} chars/token\n")
    
    # 2. Load Checkpoint BPE tốt nhất
    tokenizer = AMLBPETokenizer().load("aml_bpe_tokenizer.json")
    vocab_size = tokenizer.vocab_size
    
    model = AMLBPETransformer(vocab_size=vocab_size, max_layers=8, max_dim=256, seq_len=256, K=16, latent_dim=128).to(device)
    ckpt_path = "aml_bpe_wikitext103_best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    # 3. Đánh giá 3.000 mẫu trên cùng một giao thức kiểm thử độc lập
    num_eval = 3000
    seq_len = 256
    batch_size = 48
    n_batches = num_eval // batch_size
    crit_elem = nn.CrossEntropyLoss(reduction='none')
    
    g8_scores = []
    losses_7 = []
    losses_8 = []
    losses_soft = []
    losses_hard = []
    losses_rand = []
    
    h_curr_list = []
    h_layer8_list = []
    y_b_list = []
    
    torch.manual_seed(2026)
    np.random.seed(2026)
    
    with torch.no_grad():
        for b in range(n_batches):
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
            
            # Forward layers 1..7
            h_curr = h
            for i in range(7):
                d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
                h_layer = model.layers[i](h_curr, bm_probs[:, i, :], d_mask_i, is_first=(i==0))
                gate_i = layer_gates[:, i:i+1, None]
                h_curr = gate_i * h_layer + (1.0 - gate_i) * h_curr
                
            # Path 1: Static-7
            l7 = crit_elem(model.lm_head(model.norm_final(h_curr)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # Layer 8
            d_mask_7 = torch.sigmoid(10.0 * (d_lengths[:, 7:8, :] - arange))
            h_layer8 = model.layers[7](h_curr, bm_probs[:, 7, :], d_mask_7, is_first=False)
            
            # Path 2: Static-8
            l8 = crit_elem(model.lm_head(model.norm_final(h_layer8)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # Path 3: Soft AML
            g8 = layer_gates[:, 7:8, None]
            h_soft = g8 * h_layer8 + (1.0 - g8) * h_curr
            l_soft = crit_elem(model.lm_head(model.norm_final(h_soft)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # Path 4: Hard AML
            hard_g8 = (g8 >= 0.5).float()
            h_hard = hard_g8 * h_layer8 + (1.0 - hard_g8) * h_curr
            l_hard = crit_elem(model.lm_head(model.norm_final(h_hard)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            # Path 5: Random Skip (match P(L=8))
            p_alloc_b = (g8 >= 0.5).float().mean()
            rand_g8 = (torch.rand(B, 1, 1, device=device) < p_alloc_b).float()
            h_rand = rand_g8 * h_layer8 + (1.0 - rand_g8) * h_curr
            l_rand = crit_elem(model.lm_head(model.norm_final(h_rand)).view(-1, vocab_size), y_b.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            
            g8_scores.extend(layer_gates[:, 7].cpu().numpy())
            losses_7.extend(l7)
            losses_8.extend(l8)
            losses_soft.extend(l_soft)
            losses_hard.extend(l_hard)
            losses_rand.extend(l_rand)
            
            h_curr_list.append(h_curr.cpu())
            h_layer8_list.append(h_layer8.cpu())
            y_b_list.append(y_b.cpu())

    g8_scores = np.array(g8_scores)
    l7_arr = np.array(losses_7)
    l8_arr = np.array(losses_8)
    l_soft_arr = np.array(losses_soft)
    l_hard_arr = np.array(losses_hard)
    l_rand_arr = np.array(losses_rand)
    
    u8 = l7_arr - l8_arr # nats
    y_true_binary = (u8 > 0).astype(int)
    
    mean_g8 = np.mean(g8_scores)
    std_g8 = np.std(g8_scores)
    p_alloc = np.mean(g8_scores >= 0.5)
    rel_flops_hard = (7.0 + p_alloc) / 8.0 * 100
    rel_flops_soft = (7.0 + mean_g8) / 8.0 * 100
    
    # 4. In Bảng Ma Trận Tầng A
    print("==================== 1. ARCHITECTURE BENCHMARK MATRIX (SAME CHECKPOINT) ====================")
    print(f"{'Configuration':<28} {'Test Loss':<12} {'Subword PPL':<14} {'eBPC':<12} {'Rel FLOPs':<10} {'Status'}")
    print("-" * 90)
    print(f"{'Static 7-Layer (Truncated)':<28} {np.mean(l7_arr):<12.4f} {math.exp(np.mean(l7_arr)):<14.2f} {np.mean(l7_arr)/math.log(2)/r_exact:<12.4f} 87.5%      Baseline Floor")
    print(f"{'Random Skipping Baseline':<28} {np.mean(l_rand_arr):<12.4f} {math.exp(np.mean(l_rand_arr)):<14.2f} {np.mean(l_rand_arr)/math.log(2)/r_exact:<12.4f} {rel_flops_hard:<10.1f}% Baseline Random")
    print(f"{'Static 8-Layer (Full)':<28} {np.mean(l8_arr):<12.4f} {math.exp(np.mean(l8_arr)):<14.2f} {np.mean(l8_arr)/math.log(2)/r_exact:<12.4f} 100.0%     Full Depth")
    print(f"{'Discrete Hard-AML Router':<28} {np.mean(l_hard_arr):<12.4f} {math.exp(np.mean(l_hard_arr)):<14.2f} {np.mean(l_hard_arr)/math.log(2)/r_exact:<12.4f} {rel_flops_hard:<10.1f}% Hard Bypass")
    print(f"{'Continuous Soft-AML Router':<28} {np.mean(l_soft_arr):<12.4f} {math.exp(np.mean(l_soft_arr)):<14.2f} {np.mean(l_soft_arr)/math.log(2)/r_exact:<12.4f} {rel_flops_soft:<10.1f}% Continuous Modulation")

    # 5. Phân tích Causal Intervention
    print("\n==================== 2. CAUSAL INTERVENTION (FORCE-7 VS FORCE-8) ====================")
    mask_nat8 = (g8_scores >= 0.5)
    mask_nat7 = (g8_scores < 0.5)
    n_nat8 = np.sum(mask_nat8)
    n_nat7 = np.sum(mask_nat7)
    
    print(f"Sample Partitioning: Natural L=8 = {n_nat8:,} ({n_nat8/len(g8_scores)*100:.1f}%) | Natural L=7 = {n_nat7:,} ({n_nat7/len(g8_scores)*100:.1f}%)")
    if n_nat8 > 0:
        delta_deprived_loss = np.mean(l7_arr[mask_nat8]) - np.mean(l8_arr[mask_nat8])
        delta_deprived_ebpc = delta_deprived_loss / math.log(2) / r_exact
        t_dep, p_dep = stats.ttest_rel(l7_arr[mask_nat8], l8_arr[mask_nat8])
        print(f"  * Subgroup Natural L=8: Depriving Layer 8 -> Delta Loss = {delta_deprived_loss:+.4f} (Delta eBPC = {delta_deprived_ebpc:+.4f}, t = {t_dep:.2f}, p = {p_dep:.4e})")
    
    if n_nat7 > 0:
        delta_wasted_loss = np.mean(l8_arr[mask_nat7]) - np.mean(l7_arr[mask_nat7])
        delta_wasted_ebpc = delta_wasted_loss / math.log(2) / r_exact
        t_wasted, p_wasted = stats.ttest_rel(l8_arr[mask_nat7], l7_arr[mask_nat7])
        print(f"  * Subgroup Natural L=7: Forcing Layer 8   -> Delta Loss = {delta_wasted_loss:+.4f} (Delta eBPC = {delta_wasted_ebpc:+.4f}, t = {t_wasted:.2f}, p = {p_wasted:.4e})")

    # 6. Kiểm định CUA, Bootstrap CI, AUROC/AUPRC
    print("\n==================== 3. FORMAL CUA & DISCRIMINATION METRICS ====================")
    rho_s, p_s = stats.spearmanr(g8_scores, u8) if std_g8 > 1e-4 else (0.0, 1.0)
    rho_p, p_p = stats.pearsonr(g8_scores, u8) if std_g8 > 1e-4 else (0.0, 1.0)
    
    auroc = roc_auc_score(y_true_binary, g8_scores) if (len(np.unique(y_true_binary)) > 1 and std_g8 > 1e-4) else 0.5
    auprc = average_precision_score(y_true_binary, g8_scores) if (len(np.unique(y_true_binary)) > 1 and std_g8 > 1e-4) else 0.5
    
    n_boot = 1000
    N = len(g8_scores)
    boot_cua = [stats.spearmanr(g8_scores[idx], u8[idx])[0] for idx in [np.random.randint(0, N, N) for _ in range(n_boot)]]
    ci_low, ci_high = np.percentile(boot_cua, [2.5, 97.5])
    
    print(f"  * Gate Statistics        : Mean G_8 = {mean_g8*100:.2f}%, Std G_8 = {std_g8:.4f}")
    print(f"  * Spearman CUA_BPE (rho_S): {rho_s:+.4f} (95% Bootstrap CI: [{ci_low:+.4f}, {ci_high:+.4f}], p = {p_s:.4e})")
    print(f"  * Pearson CUA_BPE (rho_P) : {rho_p:+.4f} (p = {p_p:.4e})")
    print(f"  * AUROC (Utility > 0)     : {auroc:.4f}")
    print(f"  * AUPRC (Utility > 0)     : {auprc:.4f} (vs Random {np.mean(y_true_binary):.4f})")

    # 7. Exact Permutation Test
    print("\n==================== 4. EXACT GATE PERMUTATION TEST ====================")
    g8_perm = np.random.permutation(g8_scores)
    bpc_perm_list = []
    with torch.no_grad():
        for b in range(n_batches):
            h_c = h_curr_list[b].to(device)
            h_8 = h_layer8_list[b].to(device)
            y_ = y_b_list[b].to(device)
            g_ = torch.tensor(g8_perm[b*batch_size:(b+1)*batch_size], device=device, dtype=torch.float).view(-1, 1, 1)
            h_blend = g_ * h_8 + (1.0 - g_) * h_c
            l_s = crit_elem(model.lm_head(model.norm_final(h_blend)).view(-1, vocab_size), y_.view(-1)).view(batch_size, seq_len).mean(dim=1).cpu().numpy()
            bpc_perm_list.extend(l_s / math.log(2) / r_exact)
            
    bpc_perm = np.mean(bpc_perm_list)
    bpc_learned = np.mean(l_soft_arr) / math.log(2) / r_exact
    print(f"  * Learned Paired G_8(x) : {bpc_learned:.4f} eBPC")
    print(f"  * Permuted G_8(x_sigma) : {bpc_perm:.4f} eBPC (Delta = {bpc_perm - bpc_learned:+.4f} eBPC)")

if __name__ == '__main__':
    run_controlled_bpe_checkpoint_audit()
