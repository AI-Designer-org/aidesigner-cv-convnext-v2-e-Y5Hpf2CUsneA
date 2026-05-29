"""
ConvNeXt V2-E: Common Layers
=============================

Support modules shared across the architecture:
  - LayerNormNd: Channel-first LayerNorm for (B, C, H, W) tensors
  - DropPath: Stochastic depth for residual blocks
  - Stem: Initial patchify stem (4x4 conv, stride 4)
  - DownsampleBlock: LN -> 2x2 conv stride 2
  - ConvNeXtV2EBlock: The core block with MRF-DW + inverted bottleneck + LE-GRN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional

from mrf_dw import build_spatial_mixer
from le_grn import build_feature_competition


# ---------------------------------------------------------------------------
# LayerNormNd
# ---------------------------------------------------------------------------


class LayerNormNd(nn.Module):
    """
    Channel-first LayerNorm for (B, C, H, W) tensors.

    Equivalent to nn.LayerNorm but operating on (C, H, W) layout.
    Casts mean/variance computation to float32 for fp16/bf16 safety.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))    # (C,)
        self.bias = nn.Parameter(torch.zeros(normalized_shape))     # (C,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-first LayerNorm to (B, C, H, W) tensor.

        Mean/variance computed in float32 for fp16/bf16 safety.

        Args:
            x: (B, C, H, W) — input features.

        Returns:
            (B, C, H, W) — normalized features with learned affine transform.

        Shape invariants:
            - Input and output identical shape.
            - C must equal normalized_shape from constructor.
            - dtype in {float32, bfloat16, float16}; reduction always in float32.
        """
        B, C, H, W = x.shape                                        # (B, C, H, W)
        input_dtype = x.dtype

        # Compute in float32 for numerical stability
        x_f32 = x.float()                                           # (B, C, H, W)

        mean = x_f32.mean(dim=(1, 2, 3), keepdim=True)               # (B, 1, 1, 1)
        var = x_f32.var(dim=(1, 2, 3), keepdim=True, unbiased=False) # (B, 1, 1, 1)

        x_norm = (x_f32 - mean) / torch.sqrt(var + self.eps)         # (B, C, H, W)
        x_norm = x_norm.to(input_dtype)                             # (B, C, H, W)

        out = self.weight[:, None, None] * x_norm + self.bias[:, None, None]  # (B, C, H, W)

        return out


# ---------------------------------------------------------------------------
# DropPath
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    """
    Stochastic depth (DropPath) for residual blocks.

    Randomly drops entire samples during training with probability drop_prob.
    Scaled by 1/(1-drop_prob) at training time to maintain expected magnitude.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth: randomly drop entire samples.

        Identity in eval mode or when drop_prob=0.
        Scaled by 1/(1-drop_prob) at training to maintain expected magnitude.

        Args:
            x: (B, ...) — input tensor, any number of dimensions.

        Returns:
            (B, ...) — either dropped (zeros) or kept (scaled), same shape.

        Shape invariants:
            - Input and output identical shape.
            - Drop is sample-wise (broadcast across all non-batch dims).
            - dtype unchanged.
        """
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        # (B, 1, 1, 1) broadcast across spatial and channel dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        # floor_ is safe: random_tensor is a freshly created non-leaf tensor
        random_tensor.floor_()                                      # (B, 1, 1, 1)
        return x.div(keep_prob) * random_tensor


# ---------------------------------------------------------------------------
# Stem
# ---------------------------------------------------------------------------


class Stem(nn.Module):
    """
    ConvNeXt stem: 4x4 conv, stride 4 (patchify).

    Maps 3x224x224 -> Cx56x56 with a single strided convolution.
    No post-conv norm (matches ConvNeXt V2; norm is inside the first block).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=4, stride=4,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply patchify stem: 4x4 strided convolution.

        Args:
            x: (B, 3, H, W) — input image (typically 224x224).

        Returns:
            (B, out_channels, H/4, W/4) — stem output, norm NOT yet applied.

        Shape invariants:
            - H and W must be divisible by 4.
            - No post-conv norm; norm is inside the first block.
        """
        B, C, H, W = x.shape                                        # (B, 3, 224, 224)
        out = self.conv(x)                                          # (B, out_channels, 56, 56)
        return out


# ---------------------------------------------------------------------------
# DownsampleBlock
# ---------------------------------------------------------------------------


class DownsampleBlock(nn.Module):
    """
    ConvNeXt downsampling: LayerNorm -> Conv2D 2x2, stride 2.

    Separates normalization from spatial reduction to avoid aliasing.
    Maps (B, C_in, H, W) -> (B, C_out, H/2, W/2).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = LayerNormNd(in_channels)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=2, stride=2,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply downsampling: normalize then halve spatial resolution.

        Args:
            x: (B, C_in, H, W) — input feature map.

        Returns:
            (B, C_out, H/2, W/2) — downsampled output.

        Shape invariants:
            - H and W must be even.
            - Norm applied before spatial reduction to avoid aliasing.
        """
        B, C_in, H, W = x.shape                                     # (B, C_in, H, W)
        x = self.norm(x)                                            # (B, C_in, H, W)
        out = self.conv(x)                                          # (B, C_out, H/2, W/2)
        return out


# ---------------------------------------------------------------------------
# ConvNeXtV2EBlock
# ---------------------------------------------------------------------------


class ConvNeXtV2EBlock(nn.Module):
    """
    ConvNeXt V2-E block with MRF-DW and LE-GRN.

    Architecture:
        Input (B, C, H, W)
          ├── Spatial mixing (MRF-DW or standard 7x7 DW)
          ├── LayerNorm (pre-norm)
          ├── Conv 1x1 (expand: C -> e*C)
          ├── GELU activation
          ├── Conv 1x1 (project: e*C -> C)
          ├── Feature competition (LE-GRN or standard GRN)
          ├── DropPath
          +-- Residual connection
        Output (B, C, H, W)
    """

    def __init__(
        self,
        dim: int,
        expansion_ratio: float = 4.0,
        use_mrf_dw: bool = True,
        small_kernel_size: int = 3,
        mrf_mix_init: float = 0.0,
        base_kernel_size: int = 7,
        use_le_grn: bool = True,
        le_grn_local_kernel: int = 3,
        grn_eps: float = 1e-6,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        # ── Spatial mixing ────────────────────────────────────────────
        self.spatial_mixer = build_spatial_mixer(
            dim=dim,
            use_mrf_dw=use_mrf_dw,
            base_kernel_size=base_kernel_size,
            small_kernel_size=small_kernel_size,
            mix_init=mrf_mix_init,
        )

        # ── Channel mixing (inverted bottleneck) ──────────────────────
        self.norm = LayerNormNd(dim)                                    # (B, C, H, W)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False) # (B, C, H, W) -> (B, e*C, H, W)
        self.act = nn.GELU(approximate="tanh")                          # (B, e*C, H, W)
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False) # (B, e*C, H, W) -> (B, C, H, W)

        # ── Feature competition ───────────────────────────────────────
        self.feature_competition = build_feature_competition(
            dim=dim,
            use_le_grn=use_le_grn,
            local_kernel_size=le_grn_local_kernel,
            eps=grn_eps,
        )

        # ── Stochastic depth ──────────────────────────────────────────
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward (no checkpoint wrapping).

        Args:
            x: (B, C, H, W) — input features.

        Returns:
            (B, C, H, W) — block output, residual already added.
        """
        shortcut = x                                                    # (B, C, H, W)

        x = self.spatial_mixer(x)                                       # (B, C, H, W)
        x = self.norm(x)                                                # (B, C, H, W)
        x = self.pw1(x)                                                 # (B, C, H, W) -> (B, e*C, H, W)
        x = self.act(x)                                                 # (B, e*C, H, W)
        x = self.pw2(x)                                                 # (B, e*C, H, W) -> (B, C, H, W)
        x = self.feature_competition(x)                                 # (B, C, H, W) — residual added inside
        x = self.drop_path(x)                                           # (B, C, H, W)

        return shortcut + x                                             # (B, C, H, W)

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing.

        Args:
            x: (B, C, H, W) — input features.
            use_checkpoint: if True, use gradient checkpointing to save memory.

        Returns:
            (B, C, H, W) — block output, residual already added.

        Shape invariants:
            - Input and output identical shape.
            - Internal expansion: C -> e*C -> C.
            - dtype in {float32, bfloat16}; float16 supported via fp32 casts in LN and GRN.
        """
        if use_checkpoint and self.training:
            return checkpoint(
                self._forward, x, use_reentrant=False
            )
        return self._forward(x)
