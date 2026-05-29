"""
Multi-Receptive Field Depthwise Convolution (MRF-DW)
====================================================

Innovation 1 of ConvNeXt V2-E.

Parallel depthwise convolutions with different kernel sizes,
mixed via learnable per-channel sigmoid-gated weights.

Kernel configuration by stage:
  - Stages 1-2 (56x56, 28x28): base=7, small=3  (fine detail)
  - Stages 3-4 (14x14, 7x7):    base=7, small=5  (medium context)

Parameter overhead: ~150K (<0.8% of total model)
"""

import math
import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseSpatialOperator(ABC, nn.Module):
    """Base class for the spatial mixing operator in ConvNeXt V2-E blocks."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial mixing to input features.

        Args:
            x: (B, C, H, W) — input feature map.

        Returns:
            (B, C, H, W) — spatially mixed feature map, residual NOT yet added.

        Shape invariants:
            - Input and output identical shape.
            - H, W must be >= kernel_size.
        """
        pass


class MRFDWConv(BaseSpatialOperator):
    """
    Multi-Receptive Field Depthwise Convolution.

    Runs two parallel depthwise convolutions (base kernel + small kernel)
    and blends them with a learnable per-channel mixing weight:

        alpha = sigmoid(mix_weight)         # (1, C, 1, 1)
        out = alpha * DW_base(x) + (1-alpha) * DW_small(x)

    The mixing weight is initialized so alpha = 0.5 (equal blend).
    """

    def __init__(
        self,
        dim: int,
        base_kernel_size: int = 7,
        small_kernel_size: int = 3,
        mix_init: float = 0.0,
    ):
        super().__init__()
        self.dim = dim

        # Base kernel (7x7, matching ConvNeXt V2)
        # (B, C, H, W) -> (B, C, H, W)
        self.base_dw = nn.Conv2d(
            dim, dim,
            kernel_size=base_kernel_size,
            padding=base_kernel_size // 2,
            groups=dim,
            bias=False,
        )

        # Small kernel (3x3 in early stages, 5x5 in late stages)
        # (B, C, H, W) -> (B, C, H, W)
        small_padding = small_kernel_size // 2
        self.small_dw = nn.Conv2d(
            dim, dim,
            kernel_size=small_kernel_size,
            padding=small_padding,
            groups=dim,
            bias=False,
        )

        # Per-channel mixing weight, init to mix_init (default 0)
        # sigmoid(0) = 0.5 -> equal weighting at start
        # (1, C, 1, 1)
        self.mix_weight = nn.Parameter(
            torch.full((1, dim, 1, 1), mix_init)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale depthwise convolution with learned per-channel mixing.

        Args:
            x: (B, C, H, W) — input feature map.

        Returns:
            (B, C, H, W) — blended output, residual NOT yet added.

        Shape invariants:
            - Input and output identical shape.
            - C must equal dim passed to constructor.
            - H, W >= max(base_kernel_size, small_kernel_size).
            - dtype in {float32, bfloat16, float16}; training NaN check in native dtype.
        """
        B, C, H, W = x.shape                          # (B, C, H, W)

        base_out = self.base_dw(x)                     # (B, C, H, W)
        small_out = self.small_dw(x)                    # (B, C, H, W)

        # Sigmoid gate ensures alpha in (0, 1)
        alpha = torch.sigmoid(self.mix_weight)          # (1, C, 1, 1)

        # Per-channel convex combination
        out = alpha * base_out + (1.0 - alpha) * small_out  # (B, C, H, W)

        # Numerical safety check (training only)
        if self.training and torch.is_grad_enabled():
            assert not torch.isnan(out).any(), \
                f"NaN in {self.__class__.__name__} output"

        return out


class StandardDWConv(BaseSpatialOperator):
    """
    Standard single-kernel depthwise convolution (ConvNeXt V2 baseline).

    Used when MRF-DW is disabled (use_mrf_dw=False).
    """

    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        self.dw = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply standard 7x7 depthwise convolution.

        Args:
            x: (B, C, H, W) — input feature map.

        Returns:
            (B, C, H, W) — spatially mixed output, residual NOT yet added.
        """
        B, C, H, W = x.shape                           # (B, C, H, W)
        out = self.dw(x)                                # (B, C, H, W)

        if self.training and torch.is_grad_enabled():
            assert not torch.isnan(out).any(), \
                f"NaN in {self.__class__.__name__} output"

        return out


def build_spatial_mixer(
    dim: int,
    use_mrf_dw: bool = True,
    base_kernel_size: int = 7,
    small_kernel_size: int = 3,
    mix_init: float = 0.0,
) -> BaseSpatialOperator:
    """Factory function for the spatial mixing operator."""
    if use_mrf_dw:
        return MRFDWConv(
            dim=dim,
            base_kernel_size=base_kernel_size,
            small_kernel_size=small_kernel_size,
            mix_init=mix_init,
        )
    else:
        return StandardDWConv(dim=dim, kernel_size=base_kernel_size)
