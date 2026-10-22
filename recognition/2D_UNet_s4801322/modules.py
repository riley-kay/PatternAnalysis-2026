"""A lightweight residual 2D U-Net for multi-class OASIS MRI segmentation.

This is a proposed improved U-Net variant, not a verbatim reproduction of a
specific published architecture. Output tensors contain unnormalised logits.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)

class DownBlock(nn.Module):
    """Halve spatial dimensions, then apply a residual block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsample, concatenate encoder features, then refine."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """2D residual U-Net. Input: [B, 1, H, W]; output: [B, C, H, W]."""

    def __init__(self, in_channels: int = 1, num_classes: int = 4, base_channels: int = 32):
        super().__init__()
        if in_channels < 1 or num_classes < 2 or base_channels < 1:
            raise ValueError("in_channels >= 1, num_classes >= 2, base_channels >= 1 required")

        b = base_channels
        self.enc1 = DoubleConv(in_channels, b)
        self.enc2 = DownBlock(b, b * 2)
        self.enc3 = DownBlock(b * 2, b * 4)
        self.bottleneck = DownBlock(b * 4, b * 8)

        self.dec3 = UpBlock(b * 8, b * 4, b * 4)
        self.dec2 = UpBlock(b * 4, b * 2, b * 2)
        self.dec1 = UpBlock(b * 2, b, b)
        self.output = nn.Conv2d(b, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        x = self.bottleneck(s3)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.output(x)


if __name__ == "__main__":
    model = UNet(in_channels=1, num_classes=4, base_channels=16)
    images = torch.randn(2, 1, 129, 131)
    logits = model(images)
    print("Input:", tuple(images.shape))
    print("Output:", tuple(logits.shape))
    assert logits.shape == (2, 4, 129, 131)
    targets = torch.randint(0, 4, (2, 129, 131))
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    print("Forward/backward test passed; loss:", round(loss.item(), 4))
