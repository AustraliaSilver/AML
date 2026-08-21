import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from train_nlp_rigorous import RigorousAMLCharTransformer, evaluate_split

def train_wikitext103():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING TRUE DYNAMIC AML-CHAR ON WIKITEXT-103 ({device}) ===")
    
    data_dir = os.path.join('data', 'wikitext103')
    train_path = os.path.join(data_dir, 'train.txt')
    valid_path = os.path.join(data_dir, 'valid.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    
    # 1. Load Validation & Test sets completely into memory
    with open(valid_path, 'r', encoding='utf-8', errors='ignore') as f:
        valid_text = f.read()
    with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
        test_text = f.read()
        
    val_tensor = torch.tensor([ord(c) % 256 for c in valid_text], dtype=torch.long)
    test_tensor = torch.tensor([ord(c) % 256 for c in test_text], dtype=torch.long)
    
    # Load 10 MB slice (~10.5 million characters) of train.txt for high-speed robust training
    train_slice_size = 10 * 1024 * 1024 # 10 MB
    with open(train_path, 'r', encoding='utf-8', errors='ignore') as f:
        train_text = f.read(train_slice_size)
        
    train_tensor = torch.tensor([ord(c) % 256 for c in train_text], dtype=torch.long)
    
    print(f"\nWikiText-103 Dataset Partitions:")
    print(f"  - Training Stream (train.txt)    : {len(train_tensor):,} characters ({len(train_tensor)/(1024*1024):.2f} MB)")
    print(f"  - Validation Set (valid.txt)     : {len(val_tensor):,} characters ({len(val_tensor)/(1024*1024):.2f} MB)")
    print(f"  - Unseen Test Set (test.txt)     : {len(test_tensor):,} characters ({len(test_tensor)/(1024*1024):.2f} MB) [LOCKED]")
    
    seq_len = 128
    batch_size = 64
    epochs = 10
    steps_per_epoch = 250
    
    # 2. Build Model
    model = RigorousAMLCharTransformer(
        vocab_size=256,
        max_layers=6,
        max_dim=64,
        seq_len=seq_len,
        K=16,
        latent_dim=128
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Configuration:")
    print(f"  - Architecture  : Causal Dynamic Transformer (CausalConv1D + Causal MHA + ResNet + SwiGLU)")
    print(f"  - Total Params  : {total_params:,} (~{total_params*4/1024:.1f} KB in FP32)")
    
    optimizer = optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    best_val_bpc = float('inf')
    print(f"\nStarting 10-Epoch Training on WikiText-103 with Strict Validation Tracking:")
    
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
        
        # Validation Evaluation on actual valid.txt
        val_loss, val_bpc, val_ppl = evaluate_split(model, val_tensor, seq_len=seq_len, batch_size=batch_size, device=device)
        elapsed = time.time() - t0
        
        is_best = ""
        if val_bpc < best_val_bpc:
            best_val_bpc = val_bpc
            torch.save(model.state_dict(), 'aml_wikitext103_best.pth')
            is_best = " [BEST CHECKPOINT SAVED]"
            
        print(f"Epoch {epoch:2d}/{epochs} | {elapsed:.1f}s | Train BPC: {train_bpc:.3f} | Val BPC: {val_bpc:.3f} (Val PPL: {val_ppl:5.2f}){is_best}")
        
    # 4. FINAL UNBIASED EVALUATION ON TEST.TXT
    print("\n==================== FINAL UNBIASED TEST EVALUATION (WikiText-103 test.txt) ====================")
    model.load_state_dict(torch.load('aml_wikitext103_best.pth', map_location=device))
    test_loss, test_bpc, test_ppl = evaluate_split(model, test_tensor, seq_len=seq_len, batch_size=batch_size, device=device)
    
    print(f"Held-out WikiText-103 test.txt Evaluation Results ({len(test_tensor):,} characters):")
    print(f"  * Test Cross-Entropy Loss : {test_loss:.4f} nats/char")
    print(f"  * Test Bits-Per-Character : {test_bpc:.4f} BPC")
    print(f"  * Test Perplexity (Char)  : {test_ppl:.2f}")
    
    # 5. English Text Generation Showcase
    print("\n==================== COHERENT ENGLISH TEXT GENERATION TEST ====================")
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
            for _ in range(150):
                cur_input = input_ids[:, -seq_len:]
                logits, _ = model(cur_input, tau=0.3)
                next_char_logits = logits[0, -1, :] / 0.65 # Temperature
                probs = F.softmax(next_char_logits, dim=-1)
                next_id = torch.multinomial(probs, 1).item()
                generated += chr(next_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=-1)
                
        print(f"\n[PROMPT]   : '{prompt}'")
        print(f"[GENERATED]: '{generated}'")
        print("-" * 75)

if __name__ == '__main__':
    train_wikitext103()
