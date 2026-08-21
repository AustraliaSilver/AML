import math
import os
import time
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_wikitext103_scaled import ScaledAMLCharTransformer, set_seed if 'set_seed' in dir() else lambda s: None

def set_seed(seed=2026):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

class ScaledAMLCharTransformerWithRegularization(nn.Module):
    def __init__(self, vocab_size=256, max_layers=8, max_dim=192, seq_len=256, K=16, latent_dim=128, dropout=0.1):
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
        
        self.ctx_encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(max_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.proto_selector = nn.Linear(latent_dim, K)
        self.structure_delta = nn.Linear(latent_dim, max_layers + max_layers * max_dim)
        self.block_mode_head = nn.Linear(latent_dim, max_layers * 4)
        
        from retrain_and_verify_seed2026 import AMLScaledBlock
        self.layers = nn.ModuleList([AMLScaledBlock(max_dim, n_heads=6, dropout=dropout) for _ in range(max_layers)])
        self.norm_final = nn.LayerNorm(max_dim)
        self.lm_head = nn.Linear(max_dim, vocab_size, bias=False)
        self.lm_head.weight = self.tok_embed.weight
        
    def forward(self, x, tau=0.3):
        B, T = x.shape
        h = self.drop(self.tok_embed(x) + self.pos_embed[:, :T, :])
        z = self.ctx_encoder(h.transpose(1, 2))
        
        pi = F.softmax(self.proto_selector(z), dim=-1)
        delta = self.structure_delta(z)
        # Sửa bias khởi tạo từ +2.0 thành 0.0 để tránh bão hòa gate về 1.0
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

def train_with_sparsity_budget():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== RETRAINING WITH EXPLICIT FLOPs / SPARSITY PENALTY (cuda) ===")
    
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
    
    model = ScaledAMLCharTransformerWithRegularization().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8)
    criterion = nn.CrossEntropyLoss()
    
    batch_size = 64
    seq_len = 256
    steps_per_epoch = 350
    epochs = 8
    target_budget = 0.50 # Mục tiêu phân bổ tầng 8 ~ 50%
    lambda_budget = 0.05 # Hệ số phạt budget
    
    best_val_loss = float('inf')
    best_ckpt = 'aml_seed2026_regularized.pth'
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        g8_train_acc = 0.0
        for step in range(steps_per_epoch):
            idx = torch.randint(0, len(train_tensor) - seq_len - 1, (batch_size,))
            x = torch.stack([train_tensor[j:j+seq_len] for j in idx]).to(device)
            y = torch.stack([train_tensor[j+1:j+seq_len+1] for j in idx]).to(device)
            
            optimizer.zero_grad()
            logits, diag = model(x, tau=0.5)
            ce_loss = criterion(logits.view(-1, 256), y.view(-1))
            
            # Phạt nếu g8 vượt quá target budget (hướng router học phân biệt)
            g8 = diag['layer_gates'][:, 7]
            budget_loss = lambda_budget * ((g8.mean() - target_budget) ** 2)
            
            total_loss = ce_loss + budget_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_acc += ce_loss.item()
            g8_train_acc += g8.mean().item()
            
        scheduler.step()
        val_losses = []
        g8_val_acc = []
        model.eval()
        with torch.no_grad():
            for v_step in range(40):
                v_idx = torch.randint(0, len(valid_tensor) - seq_len - 1, (batch_size,))
                vx = torch.stack([valid_tensor[j:j+seq_len] for j in v_idx]).to(device)
                vy = torch.stack([valid_tensor[j+1:j+seq_len+1] for j in v_idx]).to(device)
                v_logits, v_diag = model(vx, tau=0.3)
                v_loss = criterion(v_logits.view(-1, 256), vy.view(-1))
                val_losses.append(v_loss.item())
                g8_val_acc.append(v_diag['layer_gates'][:, 7].mean().item())
                
        val_bpc = np.mean(val_losses) / math.log(2)
        mean_g8_val = np.mean(g8_val_acc)
        
        saved_str = ""
        if np.mean(val_losses) < best_val_loss:
            best_val_loss = np.mean(val_losses)
            torch.save(model.state_dict(), best_ckpt)
            saved_str = " -> [BEST MODEL]"
            
        print(f"Epoch {epoch:2d}/{epochs:2d} | Valid BPC: {val_bpc:.4f} | Avg Gate 8: {mean_g8_val*100:5.2f}%{saved_str}")

if __name__ == '__main__':
    train_with_sparsity_budget()
