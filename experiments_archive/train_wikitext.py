import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from AML_Core_Code.unified_aml import UnifiedAMLNetwork

def train_wikitext():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING TRUE DYNAMIC AML-CHAR MODEL ON WIKITEXT-2 ({device}) ===")
    
    # 1. Load Data
    file_path = os.path.join('data', 'wikitext2', 'test.txt')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        
    print(f"Loaded WikiText-2 dataset: {len(text):,} characters ({len(text)/1024:.1f} KB)")
    
    # Byte-level encoding (0..255)
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    
    seq_len = 128
    batch_size = 64
    vocab_size = 256
    
    # 2. Build Unified AML Model in 'text' mode
    model = UnifiedAMLNetwork(
        modality="text",
        max_layers=6,
        max_dim=48,
        vocab_size=vocab_size,
        seq_len=seq_len,
        K=16,
        latent_dim=128
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Architecture: 6 Layers Dynamic (Conv1D + ViT + ResNet + SwiGLU)")
    print(f"Total Parameters  : {total_params:,} (~{total_params*4/1024:.1f} KB in FP32)")
    
    optimizer = optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop (8 Epochs)
    epochs = 8
    steps_per_epoch = 150
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        
        # Anneal Gumbel tau
        tau = max(0.4, 2.0 - (epoch / epochs) * 1.6)
        
        for step in range(steps_per_epoch):
            # Sample random batches
            idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            x_batch = torch.stack([data[j:j+seq_len] for j in idx]).to(device)
            y_batch = torch.stack([data[j+1:j+seq_len+1] for j in idx]).to(device)
            
            optimizer.zero_grad()
            logits, diag = model(x_batch, tau=tau)
            
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
        scheduler.step()
        avg_loss = total_loss / steps_per_epoch
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch:2d}/{epochs} | {elapsed:.2f}s | Tau: {tau:.2f} | Loss: {avg_loss:.4f} | Perplexity: {ppl:6.2f}")
        
    # Save checkpoint
    torch.save(model.state_dict(), 'aml_wikitext_char.pth')
    print(f"\nSaved trained model checkpoint -> aml_wikitext_char.pth")
    
    # 4. Text Generation Showcase
    print("\n==================== TEXT GENERATION TEST ====================")
    model.eval()
    
    prompts = [
        "The tropical storm ",
        "In the history of ",
        "The military forces "
    ]
    
    for prompt in prompts:
        generated = prompt
        input_ids = torch.tensor([ord(c) % 256 for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
        
        with torch.no_grad():
            for _ in range(120):
                cur_input = input_ids[:, -seq_len:]
                logits, diag = model(cur_input, tau=0.4)
                next_char_logits = logits[0, -1, :] / 0.65 # Temperature
                probs = F.softmax(next_char_logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                
                generated += chr(next_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=-1)
                
        print(f"\n[PROMPT]   : '{prompt}'")
        print(f"[GENERATED]: '{generated}'")
        print("-" * 65)

if __name__ == '__main__':
    train_wikitext()
