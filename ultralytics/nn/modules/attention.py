import torch
import torch.nn as nn

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        ca = self.channel_att(x) * x
        sa = self.spatial_att(torch.cat([torch.mean(ca, dim=1, keepdim=True),
                                        torch.max(ca, dim=1, keepdim=True)[0]], dim=1)) * ca
        return sa