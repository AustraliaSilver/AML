import torch
import torch.nn as nn
import torch.nn.functional as F

class HtSResidualBlock(nn.Module):
    """
    Lightweight Child Residual Block receiving dynamic kernels synthesized on-the-fly.
    
    Structure:
    - Main Convolutional Path (3x3): 100% dynamic, generated on-the-fly from HtS Basis.
    - Transition Shortcut Path (1x1): Lightweight static projection used only when downsampling (stride > 1 or in_d != out_d).
    - Normalization: GroupNorm parameters.
    """
    def __init__(self, in_d, out_d, stride=1):
        super().__init__()
        self.stride = stride
        self.in_d = in_d
        self.out_d = out_d
        
        self.gn1 = nn.GroupNorm(4, in_d)
        self.gn2 = nn.GroupNorm(min(8, out_d // 4), out_d)
        
        if stride != 1 or in_d != out_d:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_d, out_d, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_d // 4), out_d)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, kernel):
        """
        Executes dynamic grouped convolution with sample-specific weights.
        Args:
            x: Input tensor [B, in_d, H, W]
            kernel: Batch-specific kernel tensor [B, out_d, in_d, 3, 3]
        """
        B, C, H, W = x.shape
        res = F.gelu(self.gn1(x))
        
        x_r = res.reshape(1, B * self.in_d, H, W)
        k_r = kernel.reshape(B * self.out_d, self.in_d, 3, 3)
        out = F.conv2d(x_r, k_r, stride=self.stride, padding=1, groups=B)
        out = out.reshape(B, self.out_d, out.shape[2], out.shape[3])
        out = self.gn2(out)
        
        return F.gelu(out + self.shortcut(x))
