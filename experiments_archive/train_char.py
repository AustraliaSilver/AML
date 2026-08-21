import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from AML_Core_Code.unified_aml import UnifiedAMLNetwork

def train_char_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== TRAINING AML CHARACTER-LEVEL SEQUENCE MODEL ON {device} ===")
    
    # 1. Training Text Corpus (Vietnamese & English mixed text sample)
    text = """Adaptive Meta-Learning (AML) la kien truc mang no-ron bien hinh tu dong.
He thong su dung 16 Prototypes va co che HtS Null-Space de hoc lien tuc ma khong bi quen (0% Catastrophic Forgetting).
Moi mau ky tu deu duoc xu ly qua 4 che do lai: CausalConv1D, ResNet, Causal Self-Attention va SwiGLU MLP.
Deep Learning 2026 voi OpenAI Triton va Dynamic HyperNetwork mang lai toc do cuc nhanh va do chinh xac vuot troi.
""" * 50 # Repeat for 200 batches
    
    # Character vocabulary
    chars = sorted(list(set(text)))
    vocab_size = 256 # Byte-level
    char_to_ix = {ch: i for i, ch in enumerate(chars)}
    ix_to_char = {i: ch for i, ch in enumerate(chars)}
    
    data = torch.tensor([ord(c) % 256 for c in text], dtype=torch.long)
    seq_len = 64
    batch_size = 32
    
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
    print(f"Model instantiated in 'text' modality: {total_params:,} parameters (~{total_params*4/1024:.1f} KB)")
    
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop (5 Epochs demo)
    epochs = 5
    num_batches = (len(data) - seq_len - 1) // (batch_size * seq_len)
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        
        for i in range(num_batches):
            # Sample batch
            idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            x_batch = torch.stack([data[j:j+seq_len] for j in idx]).to(device)
            y_batch = torch.stack([data[j+1:j+seq_len+1] for j in idx]).to(device)
            
            optimizer.zero_grad()
            logits, diag = model(x_batch, tau=0.8)
            
            # Loss over sequence: [B*T, vocab_size] vs [B*T]
            loss = criterion(logits.view(-1, vocab_size), y_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / max(1, num_batches)
        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{epochs} | {elapsed:.2f}s | CrossEntropy Loss: {avg_loss:.4f} | Perplexity: {math.exp(min(avg_loss, 20)):.2f}")
        
    # 4. Interactive Text Generation Test
    print("\n=== GENERATING SAMPLE TEXT FROM TRAINED AML-CHAR MODEL ===")
    model.eval()
    prompt = "Adaptive Meta-Learning "
    generated = prompt
    input_ids = torch.tensor([ord(c) % 256 for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(100):
            cur_input = input_ids[:, -seq_len:]
            logits, diag = model(cur_input, tau=0.5)
            next_char_logits = logits[0, -1, :] / 0.7 # Temperature scaling
            probs = F.softmax(next_char_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
            
            generated += chr(next_id)
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=-1)
            
    print(f"Prompt   : '{prompt}'")
    print(f"Generated: '{generated}'")
    print("\n[SUCCESS] Unified AML Core successfully executed Text & Character processing!")

if __name__ == '__main__':
    train_char_model()
