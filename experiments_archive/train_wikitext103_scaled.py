import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ScaledCausalDynamicBlock(nn.Module):
    """
    Higher-capacity 1D Dynamic Sequence Block with True Causal Masking:
    - Mode 0: Causal Conv1D (Kernel 3, Left-padded = 2)
    - Mode 1: ResNet-1D (Residual Pre-LN Skip)
    - Mode 2: Causal Multi-Head Attention (Strict Upper-Triangular Mask)
    - Mode 3: SwiGLU Gated MLP (Expansion factor 3x)
    """
    def __init__(self, max_dim=192, num_heads=6):
        super().__init__()
        self.max_dim = max_dim
        self.num_heads = num_heads
        self.head_dim = max_dim // num_heads
        
        # Pre-LayerNorm architecture (modern standard like LLaMA/GPT-3)
        self.ln_1 = nn.LayerNorm(max_dim)
        self.ln_2 = nn.LayerNorm(max_dim)
        
        # Mode 0: Causal Conv1D
        self.conv = nn.Conv1d(max_dim, max_dim, kernel_size=3, padding=0)
        
        # Mode 2: Causal Multi-Head Attention
        self.q_proj = nn.Linear(max_dim, max_dim)
        self.k_proj = nn.Linear(max_dim, max_dim)
        self.v_proj = nn.Linear(max_dim, max_dim)
        self.out_proj = nn.Linear(max_dim, max_dim)
        
        # Mode 3: SwiGLU Gated MLP
        hidden_dim = int(max_dim * 3.0)
        self.mlp_gate = nn.Linear(max_dim, hidden_dim)
        self.mlp_up = nn.Linear(max_dim, hidden_dim)
        self.mlp_down = nn.Linear(hidden_dim, max_dim)

    def forward(self, h, block_mode_probs, dim_mask, is_first=False):
        B, T, D = h.shape
        device = h.device
        h_norm1 = self.ln_1(h)

        # 1. Mode 0: Causal Conv1D
        h_pad = F.pad(h_norm1.transpose(1, 2), (2, 0)) # [B, D, T+2]
        h_conv = F.silu(self.conv(h_pad).transpose(1, 2)) * dim_mask

        # 2. Mode 1: ResNet Skip
        h_resnet = F.silu(h_conv) * dim_mask

        # 3. Mode 2: Causal Multi-Head Attention
        q = self.q_proj(h_norm1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_norm1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_norm1).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        scores = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)) + causal_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, T, D)
        h_attn = self.out_proj(attn_out) * dim_mask

        # Blend Attention & Conv
        g0, g1, g2, g3 = block_mode_probs[:, 0:1, None], block_mode_probs[:, 1:2, None], block_mode_probs[:, 2:3, None], block_mode_probs[:, 3:4, None]
        h_mid = h + (g0 * h_conv + g1 * h_resnet + g2 * h_attn)

        # 4. Mode 3: SwiGLU Gated Feed-Forward
        h_norm2 = self.ln_2(h_mid)
        h_mlp = self.mlp_down(F.silu(self.mlp_gate(h_norm2)) * self.mlp_up(h_norm2)) * dim_mask
        h_out = h_mid + g3 * h_mlp

        return h_out


class ScaledAMLCharTransformer(nn.Module):
    """
    Scaled 3.5M - 5M Parameter Causal Dynamic AML Character Transformer.
    """
    def __init__(self, vocab_size=256, max_layers=8, max_dim=192, seq_len=256, K=16, latent_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.seq_len = seq_len
        self.K = K
        self.latent_dim = latent_dim

        # Embeddings
        self.tok_embed = nn.Embedding(vocab_size, max_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, max_dim) * 0.02)
        self.drop = nn.Dropout(0.1)

        # Context Encoder
        self.ctx_encoder = nn.Sequential(
            nn.Conv1d(max_dim, 128, kernel_size=3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, latent_dim), nn.SiLU()
        )

        # Meta-Controllers
        self.proto_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(),
            nn.Linear(256, K)
        )
        self.structure_delta = nn.Linear(latent_dim, max_layers + (max_layers * max_dim))
        self.block_mode_head = nn.Linear(latent_dim, max_layers * 4)

        nn.init.constant_(self.structure_delta.weight, 0.0)
        nn.init.constant_(self.structure_delta.bias, 0.0)
        nn.init.constant_(self.block_mode_head.weight, 0.0)
        nn.init.constant_(self.block_mode_head.bias, 0.0)

        # 8 Scaled Dynamic Layers
        self.layers = nn.ModuleList([
            ScaledCausalDynamicBlock(max_dim=max_dim, num_heads=6)
            for _ in range(max_layers)
        ])

        self.norm_final = nn.LayerNorm(max_dim)
        self.lm_head = nn.Linear(max_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight # Weight Tying

    def forward(self, x, tau=1.0):
        B, T = x.shape
        device = x.device

        h = self.tok_embed(x) + self.pos_embed[:, :T, :]
        h = self.drop(h)

        z = self.ctx_encoder(h.transpose(1, 2))

        pi = F.softmax(self.proto_selector(z), dim=-1)
        delta = self.structure_delta(z)

        layer_gates = torch.sigmoid(delta[:, :self.max_layers] + 2.0)
        d_lengths = 64.0 + F.softplus(torch.sigmoid(delta[:, self.max_layers:].view(B, self.max_layers, self.max_dim).mean(-1, keepdim=True) + 2.0) * self.max_dim - 64.0)
        arange = torch.arange(self.max_dim, device=device).float().view(1, 1, self.max_dim)

        bm_logits = self.block_mode_head(z).view(B, self.max_layers, 4)
        bm_probs = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1)

        for i, layer in enumerate(self.layers):
            h_prev = h
            d_mask_i = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
            h_layer = layer(h, bm_probs[:, i, :], d_mask_i, is_first=(i==0))
            gate_i = layer_gates[:, i:i+1, None]
            h = gate_i * h_layer + (1.0 - gate_i) * h_prev

        h_out = self.norm_final(h)
        logits = self.lm_head(h_out)

        return logits, {
            "pi": pi,
            "layer_gates": layer_gates,
            "d_lengths": d_lengths,
            "bm_probs": bm_probs
        }


def evaluate_split(model, dataset_tensor, seq_len=256, batch_size=32, device='cuda'):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    total_chars = 0
    
    n_chunks = (len(dataset_tensor) - 1) // seq_len
    with torch.no_grad():
        for i in range(0, n_chunks, batch_size):
            actual_bs = min(batch_size, n_chunks - i)
            x_list, y_list = [], []
            for b in range(actual_bs):
                start = (i + b) * seq_len
                end = start + seq_len
                x_list.append(dataset_tensor[start:end])
                y_list.append(dataset_tensor[start+1:end+1])
                
            x_b = torch.stack(x_list).to(device)
            y_b = torch.stack(y_list).to(device)
            
            logits, _ = model(x_b, tau=0.3)
            loss = criterion(logits.view(-1, 256), y_b.view(-1))
            
            total_loss += loss.item()
            total_chars += y_b.numel()
            
    avg_loss_nats = total_loss / max(1, total_chars)
    bpc = avg_loss_nats / math.log(2)
    ppl = math.exp(min(avg_loss_nats, 20))
    return avg_loss_nats, bpc, ppl


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== SCALED AML-CHAR TRAINING ON WIKITEXT-103 ({device}) ===")
    
    data_dir = os.path.join('data', 'wikitext103')
    train_path = os.path.join(data_dir, 'train.txt')
    valid_path = os.path.join(data_dir, 'valid.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    
    with open(valid_path, 'r', encoding='utf-8', errors='ignore') as f:
        valid_text = f.read()
    with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
        test_text = f.read()
        
    val_tensor = torch.tensor([ord(c) % 256 for c in valid_text], dtype=torch.long)
    test_tensor = torch.tensor([ord(c) % 256 for c in test_text], dtype=torch.long)
    
    # Load 15 MB slice (~15.7 million characters)
    train_slice_size = 15 * 1024 * 1024
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_text = f.read(train_slice_size)
    train_tensor = torch.tensor([ord(c) % 256 for c in train_text], dtype=torch.long)
    
    print(f"Partitions:")
    print(f"  - Training Stream (train.txt)    : {len(train_tensor):,} characters ({len(train_tensor)/(1024*1024):.2f} MB)")
    print(f"  - Validation Set (valid.txt)     : {len(val_tensor):,} characters ({len(val_tensor)/(1024*1024):.2f} MB)")
    print(f"  - Held-out Test Set (test.txt)   : {len(test_tensor):,} characters ({len(test_tensor)/(1024*1024):.2f} MB) [LOCKED]")
    
    seq_len = 256
    batch_size = 32
    epochs = 12
    steps_per_epoch = 250
    
    # Instantiate Scaled Model
    model = ScaledAMLCharTransformer(
        vocab_size=256,
        max_layers=8,
        max_dim=192,
        seq_len=seq_len,
        K=16,
        latent_dim=128
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nScaled Architecture:")
    print(f"  - Hidden Dim    : 192 (up from 64) | Layers: 8 (up from 6) | Heads: 6 | Context: 256 chars")
    print(f"  - Total Params  : {total_params:,} (~{total_params*4/(1024*1024):.2f} MB in FP32)")
    
    optimizer = optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    best_val_bpc = float('inf')
    print(f"\nTraining Scaled AML-Char ({epochs} Epochs):")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        t0 = time.time()
        
        tau = max(0.4, 2.0 - (epoch / epochs) * 1.6)
        
        for step in range(steps_per_epoch):
            idx = torch.randint(0, len(train_tensor) - seq_len - 1, (batch_size,))
            x_b = torch.stack([train_tensor[j:j+seq_len] for j in idx]).to(device)
            y_b = torch.stack([train_tensor[j+1:j+seq_len+1] for j in idx]).to(device)
            
            optimizer.zero_grad()
            logits, _ = model(x_b, tau=tau)
            loss = criterion(logits.view(-1, 256), y_b.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
        scheduler.step()
        train_loss = total_train_loss / steps_per_epoch
        train_bpc = train_loss / math.log(2)
        
        val_loss, val_bpc, val_ppl = evaluate_split(model, val_tensor, seq_len=seq_len, batch_size=batch_size, device=device)
        elapsed = time.time() - t0
        
        is_best = ""
        if val_bpc < best_val_bpc:
            best_val_bpc = val_bpc
            torch.save(model.state_dict(), 'aml_scaled_wikitext103_best.pth')
            is_best = " [BEST CHECKPOINT]"
            
        print(f"Epoch {epoch:2d}/{epochs} | {elapsed:.1f}s | Train BPC: {train_bpc:.3f} | Val BPC: {val_bpc:.3f} (Val PPL: {val_ppl:5.2f}){is_best}")
        
    # FINAL TEST EVALUATION
    print("\n==================== FINAL UNBIASED TEST EVALUATION (test.txt) ====================")
    model.load_state_dict(torch.load('aml_scaled_wikitext103_best.pth', map_location=device))
    test_loss, test_bpc, test_ppl = evaluate_split(model, test_tensor, seq_len=seq_len, batch_size=batch_size, device=device)
    
    print(f"Held-out WikiText-103 test.txt Evaluation ({len(test_tensor):,} chars):")
    print(f"  * Test Cross-Entropy Loss : {test_loss:.4f} nats/char")
    print(f"  * Test Bits-Per-Character : {test_bpc:.4f} BPC")
    print(f"  * Test Perplexity (Char)  : {test_ppl:.2f}")
    
    # GENERATION
    print("\n==================== COHERENT ENGLISH TEXT GENERATION ====================")
    model.eval()
    prompts = [
        "Robert Boulter is an English ",
        "Valkyria Chronicles III is a ",
        "In the history of the eastern Atlantic "
    ]
    
    for prompt in prompts:
        generated = prompt
        input_ids = torch.tensor([ord(c) % 256 for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(180):
                cur_input = input_ids[:, -seq_len:]
                logits, _ = model(cur_input, tau=0.3)
                next_char_logits = logits[0, -1, :] / 0.65
                probs = F.softmax(next_char_logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                generated += chr(next_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=-1)
                
        print(f"\n[PROMPT]   : '{prompt}'")
        print(f"[GENERATED]: '{generated}'")
        print("-" * 80)

if __name__ == '__main__':
    main()
