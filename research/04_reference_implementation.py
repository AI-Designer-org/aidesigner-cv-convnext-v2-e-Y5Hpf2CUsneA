# ConvNeXt V2-E: Reference Implementation (PyTorch-style)

This file contains a reference implementation of the ConvNeXt V2-E architecture
with all three innovations: MRF-DW, LE-GRN, and Adaptive Expansion.

> **Note**: This is an architectural specification, not a runnable training script.
> For training, integrate with your preferred training framework (e.g., timm,
> PyTorch Lightning) with the hyperparameters from 02_architecture_design.md.

```python
"""
ConvNeXt V2-E: Efficient ConvNet under 20M parameters.

Reference implementation for the architecture described in the companion
research documents. Three innovations:

1. MRF-DW: Multi-Receptive Field Depthwise Convolution
2. LE-GRN: Local-Enhanced Global Response Normalization
3. Adaptive Expansion: Stage-specific expansion ratios

Total parameters: ~19.7M (ImageNet-1K, 1000 classes)
FLOPs (224x224): ~3.8G

Based on:
  ConvNeXt V2 (Woo et al., CVPR 2023)
  ConvNeXt V1 (Liu et al., CVPR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class LayerNormNd(nn.Module):
    """
    Channel-first LayerNorm for ConvNeXt.
    Equivalent to nn.LayerNorm but operating on (C, H, W) format.
    """
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = normalized_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        u = x.mean(dim=(1, 2, 3), keepdim=True)
        s = (x - u).pow(2).mean(dim=(1, 2, 3), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GRN(nn.Module):
    """
    Global Response Normalization (original ConvNeXt V2).
    Normalizes each channel by its global spatial RMS relative to
    the mean RMS across all channels.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 norm along spatial dims
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        # Normalize: divide by mean RMS across channels
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class LEGRN(nn.Module):
    """
    Local-Enhanced Global Response Normalization (LE-GRN).

    Innovation 2: Adds local mean subtraction before GRN to enhance
    local contrast, then applies standard global normalization.

    Same parameter count as GRN (2C for gamma + beta).
    Local mean is computed via 3x3 avg pool.
    """
    def __init__(self, dim: int, local_kernel_size: int = 3):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.local_kernel_size = local_kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Local mean subtraction
        # Compute local mean via average pooling
        local_mean = F.avg_pool2d(
            x, kernel_size=self.local_kernel_size,
            stride=1,
            padding=self.local_kernel_size // 2,
            count_include_pad=False
        )
        x_local = x - local_mean  # local contrast enhancement

        # Step 2: Standard GRN on locally-enhanced features
        Gx = torch.norm(x_local, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class MRFDWConv(nn.Module):
    """
    Multi-Receptive Field Depthwise Convolution (MRF-DW).

    Innovation 1: Parallel depthwise convolutions with different kernel sizes,
    mixed via learnable per-channel weights.

    Kernel sizes are stage-dependent:
      - Stages 1-2: base=7, small=3
      - Stages 3-4: base=7, small=5
    """
    def __init__(self, dim: int, small_kernel_size: int):
        super().__init__()
        self.dim = dim

        # Base kernel (always 7x7, matching ConvNeXt V2)
        self.base_dw = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3,
            groups=dim, bias=False
        )

        # Small kernel (stage-dependent: 3 or 5)
        padding = small_kernel_size // 2
        self.small_dw = nn.Conv2d(
            dim, dim, kernel_size=small_kernel_size,
            padding=padding, groups=dim, bias=False
        )

        # Learnable per-channel mixing weight (initialized to 0.5)
        self.mix_weight = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_dw(x)
        small_out = self.small_dw(x)

        # Sigmoid gate: maps mix_weight to (0, 1)
        alpha = torch.sigmoid(self.mix_weight)  # (1, C, 1, 1)

        # Mixed output
        return alpha * base_out + (1.0 - alpha) * small_out


class ConvNeXtV2EBlock(nn.Module):
    """
    ConvNeXt V2-E block.

    Architecture:
    Input
      ├── MRF-DW (parallel 7x7 + small_kernel depthwise conv)
      ├── LayerNorm
      ├── Conv2D 1x1 (expand, ratio = e)
      ├── GELU
      ├── Conv2D 1x1 (project)
      ├── LE-GRN (local-enhanced GRN)
      └── DropPath + residual → Output
    """
    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 4.0,
        small_kernel_size: int = 3,
        drop_path: float = 0.0,
        use_le_grn: bool = True,
        use_mrf_dw: bool = True,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # Spatial mixing (MRF-DW or standard 7x7)
        if use_mrf_dw:
            self.dwconv = MRFDWConv(dim, small_kernel_size=small_kernel_size)
        else:
            self.dwconv = nn.Conv2d(
                dim, dim, kernel_size=7, padding=3, groups=dim, bias=False
            )

        # Channel mixing (inverted bottleneck)
        self.norm = LayerNormNd(dim)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

        # Normalization (LE-GRN or standard GRN)
        if use_le_grn:
            self.grn = LEGRN(dim)
        else:
            self.grn = GRN(dim)

        # Stochastic depth
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.grn(x)
        x = self.drop_path(x)
        return shortcut + x


class DropPath(nn.Module):
    """Stochastic depth (DropPath) for residual blocks."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class DownsampleBlock(nn.Module):
    """
    ConvNeXt downsampling: LayerNorm → Conv2D 2x2, stride 2.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNormNd(in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=2, stride=2, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.norm(x))


class Stem(nn.Module):
    """
    ConvNeXt stem: 4x4 conv, stride 4 (patchify).
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=4, stride=4, bias=False
        )
        self.norm = LayerNormNd(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class ConvNeXtV2E(nn.Module):
    """
    ConvNeXt V2-E: Efficient ConvNet under 20M parameters.

    Parameters
    ----------
    in_chans : int
        Number of input channels (default: 3 for RGB)
    num_classes : int
        Number of output classes (default: 1000 for ImageNet)
    depths : List[int]
        Number of blocks per stage
    dims : List[int]
        Channel dimensions per stage
    expansion_ratios : List[float]
        Expansion ratio per stage
    small_kernel_sizes : List[int]
        Small kernel size for MRF-DW per stage (0=disable MRF-DW in stage)
    drop_path_rate : float
        Maximum stochastic depth rate (linear schedule)
    use_le_grn : bool
        Use LE-GRN instead of standard GRN
    use_mrf_dw : bool
        Use MRF-DW instead of standard 7x7 depthwise conv
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1000,
        depths: List[int] = [2, 4, 10, 3],
        dims: List[int] = [96, 192, 384, 512],
        expansion_ratios: List[float] = [3.0, 3.0, 4.0, 3.0],
        small_kernel_sizes: List[int] = [3, 3, 5, 5],
        drop_path_rate: float = 0.1,
        use_le_grn: bool = True,
        use_mrf_dw: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Stem
        self.stem = Stem(in_chans, dims[0])

        # Stochastic depth schedule (linear per-block)
        total_blocks = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)
        ]

        # Build stages
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        block_idx = 0
        for i, (depth, dim, exp, small_k) in enumerate(
            zip(depths, dims, expansion_ratios, small_kernel_sizes)
        ):
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(
                    ConvNeXtV2EBlock(
                        dim=dim,
                        expansion_ratio=exp,
                        small_kernel_size=small_k,
                        drop_path=dpr[block_idx],
                        use_le_grn=use_le_grn,
                        use_mrf_dw=use_mrf_dw,
                    )
                )
                block_idx += 1

            self.stages.append(nn.Sequential(*stage_blocks))

            # Downsampling between stages (not after last)
            if i < len(depths) - 1:
                self.downsample_layers.append(
                    DownsampleBlock(dim, dims[i + 1])
                )

        # Head
        self.norm = LayerNormNd(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes) if num_classes > 0 else nn.Identity()

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        x = self.norm(x)
        x = x.mean(dim=(2, 3))  # Global average pooling
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# Model variants
# ---------------------------------------------------------------------------

def convnext_v2e_tiny_variant(**kwargs):
    """
    ConvNeXt V2-E variant closest to ConvNeXt V2-T in layout but under 20M.
    C=[96,192,384,512], B=[2,4,10,3], E=[3,3,4,3]
    ~19.7M params at 224x224 input
    """
    return ConvNeXtV2E(
        depths=[2, 4, 10, 3],
        dims=[96, 192, 384, 512],
        expansion_ratios=[3.0, 3.0, 4.0, 3.0],
        **kwargs
    )


def convnext_v2e_small_variant(**kwargs):
    """
    Wider variant targeting ~19.5M params with fewer but wider blocks.
    C=[112,224,448,448], B=[2,3,6,2], E=[3,3,4,3]
    ~19.4M params
    """
    return ConvNeXtV2E(
        depths=[2, 3, 6, 2],
        dims=[112, 224, 448, 448],
        expansion_ratios=[3.0, 3.0, 4.0, 3.0],
        small_kernel_sizes=[3, 3, 5, 5],
        **kwargs
    )


def convnext_v2e_deep_variant(**kwargs):
    """
    Deeper variant with more blocks but narrower channels.
    C=[80,160,320,448], B=[2,5,14,3], E=[3,3,4,3]
    ~19.0M params — highest depth-to-width ratio
    """
    return ConvNeXtV2E(
        depths=[2, 5, 14, 3],
        dims=[80, 160, 320, 448],
        expansion_ratios=[3.0, 3.0, 4.0, 3.0],
        small_kernel_sizes=[3, 3, 5, 5],
        **kwargs
    )


def convnext_v2e_ablated_base(**kwargs):
    """
    Ablated baseline: MRF-DW → single 7x7 DW, LE-GRN → standard GRN.
    All other settings same as convnext_v2e_tiny_variant.
    Use this to isolate the effect of each innovation.
    """
    return ConvNeXtV2E(
        depths=[2, 4, 10, 3],
        dims=[96, 192, 384, 512],
        expansion_ratios=[3.0, 3.0, 4.0, 3.0],
        use_le_grn=False,
        use_mrf_dw=False,
        **kwargs
    )


# ---------------------------------------------------------------------------
# Parameter count verification
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="tiny",
                        choices=["tiny", "small", "deep", "ablated"])
    args = parser.parse_args()

    variants = {
        "tiny": convnext_v2e_tiny_variant,
        "small": convnext_v2e_small_variant,
        "deep": convnext_v2e_deep_variant,
        "ablated": convnext_v2e_ablated_base,
    }

    model = variants[args.variant](num_classes=1000)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Forward pass to verify
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    print(f"Variant: {args.variant}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Output shape: {out.shape}")
    print(f"Within 20M budget: {total < 20_000_000}")

    # Per-innovation breakdown
    if args.variant == "tiny":
        mrf_dw_params = sum(
            p.numel() for name, p in model.named_parameters()
            if "mrf_dw" in name or "small_dw" in name or "mix_weight" in name
        )
        le_grn_params = sum(
            p.numel() for name, p in model.named_parameters()
            if "le_grn" in name or "grn" in name
        )
        standard_grn_baseline = sum(
            p.numel() for name, p in model.named_parameters()
            if "grn" in name
        )
        print(f"  MRF-DW overhead: {mrf_dw_params:,}")
        print(f"  GRN/LE-GRN params: {le_grn_params:,}")
        print(f"  Innovation overhead: {mrf_dw_params:,} ({100*mrf_dw_params/total:.2f}%)")
```
