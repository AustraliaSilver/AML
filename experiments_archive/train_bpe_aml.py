import math
import os
import time
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from aml_tokenizer import AMLBPETokenizer

def set_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, dim_mask):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        scores = scores + causal_mask
        attn_weights = self.drop(F.softmax(scores, dim=-1))
        
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_out) * dim_mask

class AMLBPEBlock(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadCausalSelfAttention(d_model, n_heads, dropout)
        
        # SwiGLU MLP
        self.mlp_gate = nn.Linear(d_model, d_model * 3, bias=False)
        self.mlp_up = nn.Linear(d_model, d_model * 3, bias=False)
        self.mlp_down = nn.Linear(d_model * 3, d_model, bias=False)
        
        # 1D Causal Conv
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, groups=d_model)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, mode_probs, dim_mask, is_first=False):
        B, T, D = x.shape
        h_norm1 = self.norm1(x)
        h_attn = self.attn(h_norm1, dim_mask)
        
        h_norm2 = self.norm2(x + h_attn)
        h_mlp = self.mlp_down(F.silu(self.mlp_gate(h_norm2)) * self.mlp_up(h_norm2)) * dim_mask
        
        x_trans = (x + h_attn).transpose(1, 2)
        h_conv = self.conv1d(x_trans)[:, :, :T].transpose(1, 2) * dim_mask
        
        h_m0 = h_attn + h_mlp
        h_m1 = h_conv + h_mlp
        h_m2 = h_attn
        h_m3 = h_mlp
        
        h_candidates = torch.stack([h_m0, h_m1, h_m2, h_m3], dim=1)
        w = mode_probs.view(B, 4, 1, 1)
        h_selected = (h_candidates * w).sum(dim=1)
        
        return x + self.drop(h_selected)

class AMLBPETransformer(nn.Module):
    def __init__(self, vocab_size=8192, max_layers=8, max_dim=256, seq_len=256, K=16, latent_dim=128, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.seq_len = seq_len
        self.K = K
        self.latent_dim = latent_dim
        
        self.tok_embed = nn.Embedding(vocab_size, max_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, max_dim) * 0.02)
        self.drop = nn.Dropout(dropout)
        
        # Context Encoder (Lấy biểu diễn token cuối cùng thay vì pooling phẳng)
        self.ctx_encoder = nn.Sequential(
            nn.Linear(max_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.proto_selector = nn.Linear(latent_dim, K)
        self.structure_delta = nn.Linear(latent_dim, max_layers + max_layers * max_dim)
        self.block_mode_head = nn.Linear(latent_dim, max_layers * 4)
        
        self.layers = nn.ModuleList([AMLBPEBlock(max_dim, n_heads=8, dropout=dropout) for _ in range(max_layers)])
        self.norm_final = nn.LayerNorm(max_dim)
        self.lm_head = nn.Linear(max_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight # Weight Tying
        
    def forward(self, x, tau=0.3):
        B, T = x.shape
        h = self.drop(self.tok_embed(x) + self.pos_embed[:, :T, :])
        
        # Last-token context representation
        z = self.ctx_encoder(h[:, -1, :])
        
        pi = F.softmax(self.proto_selector(z), dim=-1)
        delta = self.structure_delta(z)
        
        # Khởi tạo cổng sigmoid không bias
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

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=50, temperature=0.7, top_k=40):
        self.eval()
        curr_tokens = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            x_cond = curr_tokens[:, -self.seq_len:]
            logits, _ = self(x_cond, tau=0.3)
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
        return curr_tokens

def train_bpe_aml_model():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING AML-BPE SUBWORD MODEL ON WIKITEXT-103 ({device}) ===")
    
    # 1. Load Tokenizer
    tokenizer = AMLBPETokenizer().load("aml_bpe_tokenizer.json")
    vocab_size = tokenizer.vocab_size
    print(f"Loaded Tokenizer: Vocab size = {vocab_size:,}")
    
    # 2. Tokenize & Cache Datasets
    cache_dir = "data/wikitext103_tokenized"
    os.makedirs(cache_dir, exist_ok=True)
    train_bin = os.path.join(cache_dir, "train_bpe.pt")
    valid_bin = os.path.join(cache_dir, "valid_bpe.pt")
    test_bin  = os.path.join(cache_dir, "test_bpe.pt")
    
    if not os.path.exists(train_bin):
        print("Tokenizing train.txt (15M chars)...")
        with open('data/wikitext103/train.txt', 'r', encoding='utf-8', errors='ignore') as f:
            train_text = f.read(15000000)
        train_ids = tokenizer.encode(train_text)
        torch.save(torch.tensor(train_ids, dtype=torch.long), train_bin)
        
    if not os.path.exists(valid_bin):
        print("Tokenizing valid.txt...")
        with open('data/wikitext103/valid.txt', 'r', encoding='utf-8', errors='ignore') as f:
            valid_text = f.read()
        valid_ids = tokenizer.encode(valid_text)
        torch.save(torch.tensor(valid_ids, dtype=torch.long), valid_bin)
        
    if not os.path.exists(test_bin):
        print("Tokenizing test.txt...")
        with open('data/wikitext103/test.txt', 'r', encoding='utf-8', errors='ignore') as f:
            test_text = f.read()
        test_ids = tokenizer.encode(test_text)
        torch.save(torch.tensor(test_ids, dtype=torch.long), test_bin)
        
    train_tokens = torch.load(train_bin)
    valid_tokens = torch.load(valid_bin)
    test_tokens  = torch.load(test_bin)
    
    print(f"Tokenized Data: Train = {len(train_tokens):,} tokens | Valid = {len(valid_tokens):,} tokens | Test = {len(test_tokens):,} tokens")
    
    # 3. Model Setup (D=256, 8 Layers, 8 Heads)
    model = AMLBPETransformer(vocab_size=vocab_size, max_layers=8, max_dim=256, seq_len=256, K=16, latent_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AML-BPE Model Initialized: {n_params:,} parameters (~{n_params*4/(1024*1024):.2f} MB FP32)")
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-2, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=8e-5)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 48
    seq_len = 256
    steps_per_epoch = 400
    epochs = 12
    lambda_budget = 0.015 # Phạt chi phí điện toán hướng tới ~92.7% FLOPs
    target_rel_flops = 0.927
    
    best_val_ppl = float('inf')
    best_ckpt = "aml_bpe_wikitext103_best.pth"
    
    print("\n--- STAGE 1: TRAINING ON BPE TOKENS (EARLY STOPPING ON VALID.TXT) ---")
    t0_train = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        g8_train_acc = 0.0
        
        for step in range(steps_per_epoch):
            idx = torch.randint(0, len(train_tokens) - seq_len - 1, (batch_size,))
            x = torch.stack([train_tokens[j:j+seq_len] for j in idx]).to(device)
            y = torch.stack([train_tokens[j+1:j+seq_len+1] for j in idx]).to(device)
            
            optimizer.zero_grad()
            logits, diag = model(x, tau=0.5)
            ce_loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            
            # Budget loss
            g8_batch = diag['layer_gates'][:, 7]
            expected_flops = (7.0 + g8_batch.mean()) / 8.0
            budget_loss = lambda_budget * ((expected_flops / target_rel_flops - 1.0) ** 2)
            
            total_loss = ce_loss + budget_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_acc += ce_loss.item()
            g8_train_acc += g8_batch.mean().item()
            
        scheduler.step()
        train_loss_avg = train_loss_acc / steps_per_epoch
        train_ppl = math.exp(min(train_loss_avg, 20))
        
        # Validation on valid_tokens
        model.eval()
        val_losses = []
        val_g8s = []
        with torch.no_grad():
            for v_step in range(40):
                v_idx = torch.randint(0, len(valid_tokens) - seq_len - 1, (batch_size,))
                vx = torch.stack([valid_tokens[j:j+seq_len] for j in v_idx]).to(device)
                vy = torch.stack([valid_tokens[j+1:j+seq_len+1] for j in v_idx]).to(device)
                v_logits, v_diag = model(vx, tau=0.3)
                val_losses.append(criterion(v_logits.view(-1, vocab_size), vy.view(-1)).item())
                val_g8s.append(v_diag['layer_gates'][:, 7].mean().item())
                
        val_loss_avg = float(np.mean(val_losses))
        val_ppl = math.exp(min(val_loss_avg, 20))
        mean_g8_val = float(np.mean(val_g8s))
        rel_flops_val = (7.0 + mean_g8_val) / 8.0 * 100
        
        saved_str = ""
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save(model.state_dict(), best_ckpt)
            saved_str = " -> [SAVED BEST MODEL]"
            
        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss_avg:.4f} (PPL {train_ppl:5.1f}) | Valid Loss: {val_loss_avg:.4f} (PPL {val_ppl:5.1f}) | G8: {mean_g8_val*100:4.1f}% ({rel_flops_val:4.1f}% FLOPs){saved_str}")

    print(f"\nTraining Complete in {(time.time() - t0_train)/60:.2f} mins. Best Valid PPL: {best_val_ppl:.2f}")
    
    # --- STAGE 2: FINAL EVALUATION & GENERATION ON TEST.TXT ---
    print("\n--- STAGE 2: FINAL EVALUATION & TEXT GENERATION DEMO ---")
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    
    # 1. Perplexity on Unseen test_tokens
    test_losses = []
    with torch.no_grad():
        for t_step in range(100):
            t_idx = torch.randint(0, len(test_tokens) - seq_len - 1, (batch_size,))
            tx = torch.stack([test_tokens[j:j+seq_len] for j in t_idx]).to(device)
            ty = torch.stack([test_tokens[j+1:j+seq_len+1] for j in t_idx]).to(device)
            t_logits, _ = model(tx, tau=0.3)
            test_losses.append(criterion(t_logits.view(-1, vocab_size), ty.view(-1)).item())
            
    test_loss_avg = float(np.mean(test_losses))
    test_ppl = math.exp(min(test_loss_avg, 20))
    print(f"\nFINAL UNSEEN TEST SET METRICS:")
    print(f"  * Test Cross-Entropy Loss : {test_loss_avg:.4f}")
    print(f"  * Test Word-level PPL     : {test_ppl:.2f}")

    # 2. Text Generation Demo
    prompts = [
        "Robert Boulter is an English actor who",
        "The game was developed by Sega and features",
        "In the late 19th century, the British Empire",
        "The scientific study of marine biology focuses on"
    ]
    
    print("\n--- TEXT GENERATION SHOWCASE (BPE SUBWORD MODEL) ---")
    for p in prompts:
        p_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        gen_ids = model.generate(p_ids, max_new_tokens=40, temperature=0.7, top_k=40)[0]
        gen_text = tokenizer.decode(gen_ids)
        print(f"\n[Prompt] : {p}")
        print(f"[Output] : {gen_text}")

if __name__ == '__main__':
    train_bpe_aml_model()
