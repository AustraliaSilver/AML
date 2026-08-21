import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from aml_tokenizer import AMLBPETokenizer
from modern_llama_aml import ModernLLaMAAMLTransformer, set_seed

def train_modern_llama_aml():
    set_seed(2026)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING MODERN LLAMA-STYLE AML LLM ON WIKITEXT-103 ({device}) ===")
    
    # 1. Load Tokenizer & Tokens
    tokenizer = AMLBPETokenizer().load("aml_bpe_tokenizer.json")
    vocab_size = tokenizer.vocab_size
    
    train_tokens = torch.load("data/wikitext103_tokenized/train_bpe.pt", weights_only=True)
    valid_tokens = torch.load("data/wikitext103_tokenized/valid_bpe.pt", weights_only=True)
    test_tokens  = torch.load("data/wikitext103_tokenized/test_bpe.pt", weights_only=True)
    
    print(f"Loaded Datasets: Train={len(train_tokens):,} | Valid={len(valid_tokens):,} | Test={len(test_tokens):,} tokens")
    
    # 2. Setup Modern LLaMA-style AML LLM
    # max_layers=10, max_dim=384, intermediate_dim=1024, seq_len=512
    model = ModernLLaMAAMLTransformer(
        vocab_size=vocab_size,
        max_layers=10,
        max_dim=384,
        intermediate_dim=1024,
        seq_len=512,
        K=16,
        latent_dim=128
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model Initialized: {n_params:,} parameters (~{n_params*4/(1024*1024):.2f} MB FP32) | Context Length: {model.seq_len} tokens")
    
    # 3. Optimizer with AdamW + Cosine Annealing + Weight Decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-2, betas=(0.9, 0.95), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=5e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Enable AMP (Automatic Mixed Precision - FP16) for fast & memory-efficient RTX 3050 training
    scaler = torch.cuda.amp.GradScaler()
    
    batch_size = 24
    seq_len = 512
    steps_per_epoch = 300
    epochs = 8
    
    best_val_loss = float('inf')
    best_ckpt = "modern_llama_aml_wikitext103_best.pth"
    
    print("\n--- STAGE 1: TRAINING ON WIKITEXT-103 (EARLY STOPPING ON VALID.TXT) ---")
    t0_train = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_acc = 0.0
        g_acc = 0.0
        t0_epoch = time.time()
        
        for step in range(steps_per_epoch):
            idx = torch.randint(0, len(train_tokens) - seq_len - 1, (batch_size,))
            x = torch.stack([train_tokens[j:j+seq_len] for j in idx]).to(device)
            y = torch.stack([train_tokens[j+1:j+seq_len+1] for j in idx]).to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits, diag = model(x, tau=0.5)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_acc += loss.item()
            g_acc += diag['layer_gates'].mean().item()
            
        scheduler.step()
        train_loss_avg = train_loss_acc / steps_per_epoch
        train_ppl = math.exp(min(train_loss_avg, 20))
        
        # Validation on valid_tokens
        model.eval()
        val_losses = []
        with torch.no_grad():
            for v_step in range(25):
                v_idx = torch.randint(0, len(valid_tokens) - seq_len - 1, (batch_size,))
                vx = torch.stack([valid_tokens[j:j+seq_len] for j in v_idx]).to(device)
                vy = torch.stack([valid_tokens[j+1:j+seq_len+1] for j in v_idx]).to(device)
                with torch.cuda.amp.autocast():
                    v_logits, _ = model(vx, tau=0.3)
                    v_loss = criterion(v_logits.view(-1, vocab_size), vy.view(-1))
                val_losses.append(v_loss.item())
                
        val_loss = float(np.mean(val_losses))
        val_ppl = math.exp(min(val_loss, 20))
        mean_g_val = g_acc / steps_per_epoch
        
        saved_str = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)
            saved_str = " -> [SAVED BEST MODEL]"
            
        elapsed_epoch = time.time() - t0_epoch
        print(f"Epoch {epoch:2d}/{epochs:2d} ({elapsed_epoch/60:.1f}m) | Train Loss: {train_loss_avg:.4f} (PPL {train_ppl:5.1f}) | Valid Loss: {val_loss:.4f} (PPL {val_ppl:5.1f}) | Avg Gates: {mean_g_val*100:4.1f}%{saved_str}")

    print(f"\nTraining Complete in {(time.time() - t0_train)/60:.2f} mins. Best Valid PPL: {math.exp(min(best_val_loss, 20)):.2f}")
    
    # --- STAGE 2: EVALUATION ON UNSEEN TEST SET & TEXT GENERATION ---
    print("\n--- STAGE 2: EVALUATION ON UNSEEN TEST SET & TEXT GENERATION ---")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()
    
    test_losses = []
    with torch.no_grad():
        for t_step in range(60):
            t_idx = torch.randint(0, len(test_tokens) - seq_len - 1, (batch_size,))
            tx = torch.stack([test_tokens[j:j+seq_len] for j in t_idx]).to(device)
            ty = torch.stack([test_tokens[j+1:j+seq_len+1] for j in t_idx]).to(device)
            with torch.cuda.amp.autocast():
                t_logits, _ = model(tx, tau=0.3)
                t_loss = criterion(t_logits.view(-1, vocab_size), ty.view(-1))
            test_losses.append(t_loss.item())
            
    test_loss_avg = float(np.mean(test_losses))
    test_ppl = math.exp(min(test_loss_avg, 20))
    print(f"FINAL UNSEEN TEST METRICS:")
    print(f"  * Test Cross-Entropy Loss : {test_loss_avg:.4f}")
    print(f"  * Test Perplexity (PPL)   : {test_ppl:.2f}")
    
    # Text Generation Showcase
    prompts = [
        "Robert Boulter is an English actor who starred in",
        "The game was developed by Sega and features dynamic",
        "In the late 19th century, the British Empire expanded into",
        "The scientific study of marine biology focuses on ocean organisms and"
    ]
    
    print("\n--- MODERN LLAMA-AML TEXT GENERATION SHOWCASE ---")
    for p in prompts:
        p_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        gen_ids = model.generate(p_ids, max_new_tokens=50, temperature=0.7, top_k=40, repetition_penalty=1.15)[0]
        gen_text = tokenizer.decode(gen_ids)
        print(f"\n[Prompt]: {p}")
        print(f"[Output]: {gen_text}")

if __name__ == '__main__':
    train_modern_llama_aml()
