import torch
import torch.nn.functional as F
from train_wikitext103_scaled import ScaledAMLCharTransformer

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ScaledAMLCharTransformer(vocab_size=256, max_layers=8, max_dim=192, seq_len=256, K=16, latent_dim=128).to(device)
    model.load_state_dict(torch.load('aml_scaled_wikitext103_best.pth', map_location=device))
    model.eval()

    segments = {
        'Segment A (Biography / History)': 'Robert Boulter is an English film, television and theatre actor who had a guest-starring role in 2000.',
        'Segment B (Video Game / Sci-Fi)': 'Valkyria Chronicles III is a tactical role-playing video game developed by Sega and Media.Vision for PSP.',
        'Segment C (Biology / Oceanography)': 'Homarus gammarus, known as the European lobster, is a species of clawed lobster from the eastern Atlantic Ocean.',
        'Segment D (Math / Python Code)': 'def compute_tensor_grad(alpha, beta, gamma): return torch.matmul(alpha, beta) + gamma * 3.14159',
        'Segment E (Poetry / Literature)': 'O wind, if winter comes, can spring be far behind? The pale shadows lengthen across the quiet winter snow.'
    }

    mode_names = ['CausalConv1D', 'ResNet-1D', 'Causal-ViT', 'SwiGLU-MLP']

    print('==================== DYNAMIC ARCHITECTURE TRAJECTORY INFERENCE TEST ====================')
    for name, text in segments.items():
        input_ids = torch.tensor([ord(c) % 256 for c in text[:256]], dtype=torch.long).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, diag = model(input_ids, tau=0.3)
            
        layer_gates = diag['layer_gates'][0]
        d_lengths = diag['d_lengths'][0].view(8)
        bm_probs = diag['bm_probs'][0] # [8, 4]
        pi = diag['pi'][0]
        
        active_layers = (layer_gates > 0.5).sum().item()
        mean_d = d_lengths.mean().item()
        
        trajectory = []
        for i in range(8):
            m_idx = bm_probs[i].argmax().item()
            p_val = bm_probs[i][m_idx].item()
            trajectory.append(f'{mode_names[m_idx]}[{p_val*100:.0f}%]')
            
        top_pi = pi.topk(3)
        pi_str = ', '.join([f'P#{idx+1}:{val*100:.1f}%' for idx, val in zip(top_pi.indices.tolist(), top_pi.values.tolist())])
        
        print(f'\n>>> {name}:')
        print(f'  * Text Sample    : \"{text[:70]}...\"')
        print(f'  * Dynamic Depth  : L = {active_layers}/8 layers (Gates: {[round(g.item(), 2) for g in layer_gates]})')
        print(f'  * Dynamic Width  : D_mean = {mean_d:.1f}/192 channels (Per-layer D: {[round(d.item(), 1) for d in d_lengths]})')
        print(f'  * Prototype Blend: {pi_str}')
        print(f'  * Mode Sequence  : ' + ' -> '.join(trajectory[:active_layers]))

if __name__ == '__main__':
    main()
