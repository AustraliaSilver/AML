import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from aml_tokenizer import AMLBPETokenizer
from train_bpe_aml import AMLBPETransformer

def evaluate_bpe_checkpoint():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== EVALUATING AML-BPE TRAINED CHECKPOINT ({device}) ===")
    
    tokenizer = AMLBPETokenizer().load("aml_bpe_tokenizer.json")
    vocab_size = tokenizer.vocab_size
    
    # Load model
    model = AMLBPETransformer(vocab_size=vocab_size, max_layers=8, max_dim=256, seq_len=256, K=16, latent_dim=128).to(device)
    ckpt_path = "aml_bpe_wikitext103_best.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    
    # 1. Evaluate on Unseen test.txt
    test_bin = "data/wikitext103_tokenized/test_bpe.pt"
    test_tokens = torch.load(test_bin)
    
    criterion = nn.CrossEntropyLoss()
    test_losses = []
    batch_size = 48
    seq_len = 256
    
    with torch.no_grad():
        for t_step in range(100):
            t_idx = torch.randint(0, len(test_tokens) - seq_len - 1, (batch_size,))
            tx = torch.stack([test_tokens[j:j+seq_len] for j in t_idx]).to(device)
            ty = torch.stack([test_tokens[j+1:j+seq_len+1] for j in t_idx]).to(device)
            t_logits, _ = model(tx, tau=0.3)
            test_losses.append(criterion(t_logits.view(-1, vocab_size), ty.view(-1)).item())
            
    test_loss_avg = float(torch.tensor(test_losses).mean().item())
    test_ppl = math.exp(min(test_loss_avg, 20))
    
    print("\n--- 1. UNSEEN TEST SET METRICS ---")
    print(f"  * Cross-Entropy Loss : {test_loss_avg:.4f}")
    print(f"  * Subword Token PPL  : {test_ppl:.2f}")
    print(f"  * Equivalent BPC     : {test_loss_avg / math.log(2) / 4.14:.4f} bits/char (Chuẩn hóa về char-level)")

    # 2. Text Generation Showcase
    prompts = [
        "Robert Boulter is an English actor who",
        "The game was developed by Sega and features",
        "In the late 19th century, the British Empire",
        "The scientific study of marine biology focuses on",
        "Valkyria Chronicles III is a tactical role-playing video game"
    ]
    
    print("\n--- 2. TEXT GENERATION SHOWCASE (BPE SUBWORD MODEL) ---")
    for p in prompts:
        p_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        gen_ids = model.generate(p_ids, max_new_tokens=45, temperature=0.7, top_k=40)[0]
        gen_text = tokenizer.decode(gen_ids)
        print(f"\n[Prompt] : {p}")
        print(f"[Output] : {gen_text}")

if __name__ == '__main__':
    evaluate_bpe_checkpoint()
