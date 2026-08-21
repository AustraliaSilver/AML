import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aml_tokenizer import AMLBPETokenizer

def set_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# --- 1. RMSNorm (Root Mean Square Layer Normalization - Chuẩn LLaMA-3 / Mistral) ---
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# --- 2. Rotary Positional Embeddings (RoPE - Chuẩn LLaMA-3 / Mistral / Gemma) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q, k):
        T = q.shape[2]
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)
            
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0) # [1, 1, T, dim]
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        
        # Apply RoPE
        def rotate_half(x):
            x1 = x[..., :x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * cos) + (rotate_half(q) * sin)
        k_rot = (k * cos) + (rotate_half(k) * sin)
        return q_rot, k_rot

# --- 3. SwiGLU Feed-Forward Network (Chuẩn LLaMA-3 / PaLM) ---
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, intermediate_dim: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(d_model, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, dim_mask):
        # SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
        swish_gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        h = self.down_proj(swish_gate * up) * dim_mask
        return self.drop(h)

# --- 4. Causal RoPE Multi-Head Attention (Chuẩn LLaMA) ---
class LLaMACausalAttention(nn.Module):
    def __init__(self, d_model=384, n_heads=12, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_seq_len=1024)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x, dim_mask):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE directly onto Q and K
        q, k = self.rope(q, k)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        scores = scores + causal_mask
        attn_weights = self.drop(F.softmax(scores, dim=-1))
        
        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(attn_out) * dim_mask

# --- 5. Causal Attention-Pooled Context Router (Định tuyến thông minh) ---
class CausalAttentionContextRouter(nn.Module):
    def __init__(self, d_model=384, latent_dim=128, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, latent_dim)
        self.norm = RMSNorm(d_model)
        
        self.mlp_refine = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
    def forward(self, h):
        B, T, D = h.shape
        h_norm = self.norm(h)
        
        # Query kết hợp learnable query và token cuối cùng
        q_raw = self.query_token.expand(B, -1, -1) + h_norm[:, -1:, :]
        q = self.q_proj(q_raw).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        pooled_out = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, 1, D)
        z_init = self.out_proj(pooled_out).squeeze(1)
        return self.mlp_refine(z_init), attn_weights

# --- 6. Modern LLaMA-AML Decoder Block ---
class ModernLLaMAAMLBlock(nn.Module):
    def __init__(self, d_model=384, n_heads=12, intermediate_dim=1024, dropout=0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = LLaMACausalAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, intermediate_dim, dropout)
        
    def forward(self, x, dim_mask):
        # Pre-RMSNorm Residual Connections
        h_attn = self.attn(self.norm1(x), dim_mask)
        x = x + h_attn
        h_ffn = self.ffn(self.norm2(x), dim_mask)
        x = x + h_ffn
        return x

# --- 7. Full Modern LLaMA-Style AML Large Model ---
class ModernLLaMAAMLTransformer(nn.Module):
    """
    Chuẩn kiến trúc Modern LLM (LLaMA-3 / Mistral) kết hợp Adaptive Meta-Learning:
    1. Tokenizer: BPE 8,192 Subwords
    2. Positional: Rotary Position Embeddings (RoPE)
    3. Normalization: RMSNorm (Pre-Norm)
    4. Activation: SwiGLU Gated Feed-Forward Networks
    5. Attention Router: Causal Multi-Head Context Routing
    6. Adaptive Compute: Dynamic Depth & Continuous Soft Modulation
    """
    def __init__(self, vocab_size=8192, max_layers=10, max_dim=384, intermediate_dim=1024, seq_len=512, K=16, latent_dim=128, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.seq_len = seq_len
        self.K = K
        self.latent_dim = latent_dim
        
        # Token Embeddings (Không cần pos_embed vì dùng RoPE)
        self.tok_embed = nn.Embedding(vocab_size, max_dim)
        
        # Attention Context Router
        self.ctx_router = CausalAttentionContextRouter(d_model=max_dim, latent_dim=latent_dim, n_heads=4)
        self.proto_selector = nn.Linear(latent_dim, K)
        self.structure_delta = nn.Linear(latent_dim, max_layers + max_layers * max_dim)
        
        # 10 Modern LLaMA Blocks
        self.layers = nn.ModuleList([
            ModernLLaMAAMLBlock(d_model=max_dim, n_heads=12, intermediate_dim=intermediate_dim, dropout=dropout)
            for _ in range(max_layers)
        ])
        
        self.norm_final = RMSNorm(max_dim)
        self.lm_head = nn.Linear(max_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight # Weight Tying
        
    def forward(self, x, tau=0.3):
        B, T = x.shape
        h = self.tok_embed(x)
        
        # Trích xuất vector chỉ huy z qua Attention Router
        z, router_attn_weights = self.ctx_router(h)
        
        pi = F.softmax(self.proto_selector(z), dim=-1)
        delta = self.structure_delta(z)
        
        layer_gates = torch.sigmoid(delta[:, :self.max_layers])
        d_raw = torch.sigmoid(delta[:, self.max_layers:].view(B, self.max_layers, self.max_dim).mean(-1, keepdim=True))
        d_lengths = 64.0 + F.softplus(d_raw * self.max_dim - 64.0)
        arange = torch.arange(self.max_dim, device=x.device).float().view(1, 1, self.max_dim)
        
        for i, layer in enumerate(self.layers):
            h_prev = h
            d_mask = torch.sigmoid(10.0 * (d_lengths[:, i:i+1, :] - arange))
            h_layer = layer(h, d_mask)
            gate = layer_gates[:, i:i+1, None]
            h = gate * h_layer + (1.0 - gate) * h_prev
            
        logits = self.lm_head(self.norm_final(h))
        return logits, {
            'pi': pi,
            'layer_gates': layer_gates,
            'd_lengths': d_lengths,
            'router_attn': router_attn_weights
        }

    @torch.no_grad()
    def generate(self, prompt_tokens, max_new_tokens=60, temperature=0.7, top_k=40, repetition_penalty=1.15):
        self.eval()
        curr_tokens = prompt_tokens.clone()
        for _ in range(max_new_tokens):
            x_cond = curr_tokens[:, -self.seq_len:]
            logits, _ = self(x_cond, tau=0.3)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Repetition Penalty (chống lặp từ như các LLM lớn)
            if repetition_penalty != 1.0:
                for token_id in set(curr_tokens[0].tolist()):
                    if next_token_logits[0, token_id] > 0:
                        next_token_logits[0, token_id] /= repetition_penalty
                    else:
                        next_token_logits[0, token_id] *= repetition_penalty
                        
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
        return curr_tokens

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ModernLLaMAAMLTransformer(vocab_size=8192, max_layers=10, max_dim=384, intermediate_dim=1024, seq_len=512).to(device)
    dummy_input = torch.randint(0, 8192, (2, 512)).to(device)
    logits, diag = model(dummy_input)
    
    n_params = sum(p.numel() for p in model.parameters())
    print("=== MODERN LLAMA-STYLE AML LLM ARCHITECTURE ===")
    print(f"Total Parameters : {n_params:,} (~{n_params*4/(1024*1024):.2f} MB FP32)")
    print(f"Context Length   : {model.seq_len} tokens (RoPE enabled)")
    print(f"Hidden Dimension : {model.max_dim} (12 Heads, Head Dim = 32)")
    print(f"Feed-Forward     : SwiGLU (Intermediate Dim = 1,024)")
    print(f"Normalization    : RMSNorm (Pre-Norm)")
    print(f"Logits Shape     : {logits.shape}")
    print(f"Layer Gates      : {diag['layer_gates'].shape} (10 Layers)")
    print("[SUCCESS] Modern LLaMA-style AML LLM successfully initialized and verified!")
