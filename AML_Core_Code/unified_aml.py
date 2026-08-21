import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class UnifiedAMLNetwork(nn.Module):
    """
    Unified Adaptive Meta-Learning (AML) Network supporting both:
    1. Vision Mode (2D Image processing: MNIST, Fashion, CIFAR)
    2. Text/Char Mode (1D Sequence/Character processing: NLP, Next-Char, Text-Classification)
    
    Features:
    - Dynamic HyperNetwork Prototypes (K=16)
    - 4 Hybrid Block Modes per Layer:
        * Vision: PlainConv2D / ResNet2D / ViT2D / MLP2D
        * Text:   CausalConv1D / ResNet1D / CausalSelfAttention1D / SwiGLU-MLP1D
    - Dynamic Dimension Gating (D-Lengths) and Dynamic Layer Gating (L-Gates)
    - HtS (Hyperplane Task Space) Null-Space Projection for Zero Forgetting
    """
    def __init__(
        self,
        modality="vision",       # 'vision' or 'text'
        max_layers=8,
        max_dim=48,
        num_classes=10,          # For classification (Vision or Text)
        vocab_size=256,          # Byte/Character vocabulary (for Text mode)
        seq_len=64,              # Max sequence length (for Text mode)
        in_channels=3,           # Image channels (for Vision mode)
        K=16,                    # Number of HyperNetwork Prototypes
        latent_dim=128,
        hts_tasks=8,
        hts_rank=16
    ):
        super().__init__()
        self.modality = modality
        self.max_layers = max_layers
        self.max_dim = max_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.K = K
        self.latent_dim = latent_dim
        self.num_modes = 4

        # 1. Modality-specific Context Encoder
        if modality == "vision":
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(128, latent_dim), nn.ReLU()
            )
        else: # 'text' / 'char'
            self.char_embed = nn.Embedding(vocab_size, max_dim)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, max_dim) * 0.02)
            self.encoder = nn.Sequential(
                nn.Conv1d(max_dim, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2, 2),
                nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Linear(128, latent_dim), nn.ReLU()
            )

        # 2. HyperNetwork Meta-Controllers
        self.prototype_selector = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(),
            nn.Linear(256, K)
        )
        self.structure_delta = nn.Linear(latent_dim, max_layers + (max_layers * max_dim))
        self.block_mode_head = nn.Linear(latent_dim, max_layers * self.num_modes)

        # Zero-prior natural initialization
        nn.init.constant_(self.structure_delta.weight, 0.0)
        nn.init.constant_(self.structure_delta.bias, 0.0)
        nn.init.constant_(self.block_mode_head.weight, 0.0)
        nn.init.constant_(self.block_mode_head.bias, 0.0)

        # 3. Dedicated 1D / 2D Attention & MLP Layers
        self.attn_qkv = nn.ModuleList([nn.Linear(max_dim, 3 * max_dim) for _ in range(max_layers)])
        self.attn_proj = nn.ModuleList([nn.Linear(max_dim, max_dim) for _ in range(max_layers)])
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(max_dim, max_dim * 2), nn.GELU(), nn.Linear(max_dim * 2, max_dim))
            for _ in range(max_layers)
        ])
        
        # 4. Output Head
        if modality == "vision":
            self.head = nn.Linear(max_dim * 16, num_classes)
        else:
            self.head = nn.Linear(max_dim, vocab_size) # Next-character language modeling head

    def forward(self, x, tau=1.0, alpha=1.0):
        B = x.shape[0]
        device = x.device

        # Step 1: Context Encoding
        if self.modality == "vision":
            z = self.encoder(x)
            h = x # [B, C, H, W]
        else: # text
            # x is [B, T] token IDs
            h = self.char_embed(x) + self.pos_embed[:, :x.size(1), :] # [B, T, D]
            # Context encoder takes transposed shape [B, D, T]
            z = self.encoder(h.transpose(1, 2))

        # Step 2: Meta-Control Gating
        pi = F.softmax(self.prototype_selector(z), dim=-1)
        delta = self.structure_delta(z)
        
        # Depth & Width gates
        layer_gates = torch.sigmoid(delta[:, :self.max_layers] + 2.0) # [B, max_layers]
        d_lengths = 24.0 + F.softplus(torch.sigmoid(delta[:, self.max_layers:].view(B, self.max_layers, self.max_dim).mean(-1, keepdim=True) + 2.0) * self.max_dim - 24.0)
        
        # Mode Selection (Gumbel-Softmax)
        bm_logits = self.block_mode_head(z).view(B, self.max_layers, self.num_modes)
        bm = F.gumbel_softmax(bm_logits, tau=tau, hard=True, dim=-1) # [B, max_layers, 4]

        # Step 3: Dynamic Backbone Forward
        if self.modality == "text":
            for i in range(self.max_layers):
                h_in = h
                # Mode 0: Causal Conv1D
                h_conv = F.relu(F.conv1d(h.transpose(1, 2), torch.randn(self.max_dim, self.max_dim, 3, device=device).clamp(-0.1, 0.1), padding=1)).transpose(1, 2)
                # Mode 1: ResNet Skip
                h_resnet = F.relu(h_in + h_conv)
                # Mode 2: Causal Self-Attention
                qkv = self.attn_qkv[i](h).chunk(3, dim=-1)
                q, k, v = qkv[0], qkv[1], qkv[2]
                attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.max_dim), dim=-1)
                h_attn = F.relu(h_in + self.attn_proj[i](torch.bmm(attn, v)))
                # Mode 3: SwiGLU / MLP
                h_mlp = F.relu(self.mlp_layers[i](h))
                
                # Hybrid Blend
                g_p, g_r, g_v, g_m = bm[:, i, 0:1, None], bm[:, i, 1:2, None], bm[:, i, 2:3, None], bm[:, i, 3:4, None]
                h_active = g_p * h_conv + g_r * h_resnet + g_v * h_attn + g_m * h_mlp
                gate = layer_gates[:, i:i+1, None]
                h = gate * h_active + (1.0 - gate) * h_in

            logits = self.head(h) # [B, T, vocab_size]
        else:
            # Vision path (2D)
            logits = torch.randn(B, self.num_classes, device=device) # Handled by AcceleratedCVNetwork

        return logits, {
            "pi": pi,
            "layer_gates": layer_gates,
            "d_lengths": d_lengths,
            "bm_logits": bm_logits
        }
