import os
import math
import sys
import torch
import torch.nn as nn
from aml_tokenizer import AMLBPETokenizer
from train_bpe_aml import AMLBPETransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AMLBPETokenizer().load("aml_bpe_tokenizer.json")
vocab_size = tokenizer.vocab_size

model = AMLBPETransformer(vocab_size=vocab_size, max_layers=8, max_dim=256, seq_len=256, K=16, latent_dim=128).to(device)
ckpt_path = "aml_bpe_wikitext103_best.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
model.eval()

# Test PPL
test_tokens = torch.load("data/wikitext103_tokenized/test_bpe.pt", weights_only=True)
criterion = nn.CrossEntropyLoss()
losses = []
batch_size = 32
with torch.no_grad():
    for t_step in range(100):
        t_idx = torch.randint(0, len(test_tokens) - 256 - 1, (batch_size,))
        tx = torch.stack([test_tokens[j:j+256] for j in t_idx]).to(device)
        ty = torch.stack([test_tokens[j+1:j+256+1] for j in t_idx]).to(device)
        t_logits, _ = model(tx, tau=0.3)
        losses.append(criterion(t_logits.view(-1, vocab_size), ty.view(-1)).item())

avg_loss = sum(losses) / len(losses)
ppl = math.exp(avg_loss)
bpc_equiv = avg_loss / math.log(2) / 4.14

with open("eval_results.txt", "w", encoding="utf-8") as f:
    f.write(f"TEST LOSS: {avg_loss:.4f}\n")
    f.write(f"TEST SUBWORD PPL: {ppl:.2f}\n")
    f.write(f"EQUIVALENT BPC: {bpc_equiv:.4f}\n\n")
    
    prompts = [
        "Robert Boulter is an English actor who",
        "The game was developed by Sega and features",
        "In the late 19th century, the British Empire",
        "The scientific study of marine biology focuses on"
    ]
    
    for p in prompts:
        p_ids = torch.tensor([tokenizer.encode(p)], dtype=torch.long, device=device)
        gen_ids = model.generate(p_ids, max_new_tokens=40, temperature=0.7, top_k=40)[0]
        gen_text = tokenizer.decode(gen_ids)
        f.write(f"[Prompt]: {p}\n")
        f.write(f"[Output]: {gen_text}\n\n")

print("EVALUATION FINISHED SUCCESSFULLY! Saved to eval_results.txt")
