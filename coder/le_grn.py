"""
Local-Enhanced Global Response Normalization (LE-GRN)
======================================================

Innovation 2 of ConvNeXt V2-E.

Augments GRN with a local mean subtraction step before global normalization.

Standard GRN:
    Gx = ||x||_2 over (H, W)       -- global RMS per channel
    Nx = Gx / mean(Gx)             -- normalize across channels
    out = gamma * (x * Nx) + beta + x

LE-GRN:
    x_local = x - mu_local(x)      -- local mean removal (3x3 avg pool)
    Gx = ||x_local||_2             -- global RMS of contrast-enhanced features
    Nx = Gx / mean(Gx)             -- normalize across channels
    out = gamma * (x * Nx) + beta + x

Zero additional parameters vs standard GRN (both use 2C for gamma + beta).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseFeatureCompetition(ABC, nn.Module):
    """Base class for feature competition/normalization after channel mixing."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature competition normalization.

        Args:
            x: (B, C, H, W) — input feature map (post pointwise projection).

        Returns:
            (B, C, H, W) — normalized feature map, residual already added inside module.
        """
        pass


class LEGRN(BaseFeatureCompetition):
    """
    Local-Enhanced Global Response Normalization.

    Adds local contrast enhancement (mean subtraction via 3x3 avg pool)
    before standard GRN. Zero additional parameters vs GRN.

    Numerical safety: L2 norm computation is cast to float32 to avoid
    fp16/bf16 overflow for channels with large activations.
    """

    def __init__(
        self,
        dim: int,
        local_kernel_size: int = 3,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))    # (1, C, 1, 1)
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))     # (1, C, 1, 1)
        self.local_kernel_size = local_kernel_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local-enhanced global response normalization.

        Local mean subtraction before standard GRN. Residual is added inside.

        Args:
            x: (B, C, H, W) — input feature map (post pointwise projection).

        Returns:
            (B, C, H, W) — normalized feature map, residual already added.

        Shape invariants:
            - Input and output identical shape.
            - L2 norm computed in float32 to prevent fp16/bf16 overflow.
            - dtype in {float32, bfloat16, float16}.
        """
        B, C, H, W = x.shape                                   # (B, C, H, W)
        input_dtype = x.dtype

        # Step 1: Local mean subtraction (contrast enhancement)
        # 3x3 avg pool with padding=1 preserves spatial dimensions
        local_mean = F.avg_pool2d(
            x,
            kernel_size=self.local_kernel_size,
            stride=1,
            padding=self.local_kernel_size // 2,
            count_include_pad=False,
        )                                                       # (B, C, H, W)
        x_local = x - local_mean                                # (B, C, H, W)

        # Step 2: Global RMS of locally-contrasted features
        # Cast to float32 for numerical safety (L2 norm can overflow in fp16)
        Gx = torch.norm(
            x_local.float(), p=2, dim=(2, 3), keepdim=True
        )                                                       # (B, C, 1, 1)

        # Step 3: Normalize across channels
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)    # (B, C, 1, 1)
        Nx = Nx.to(input_dtype)

        # Step 4: Element-wise scaling + residual
        # Note: scaling uses original x (not x_local) to preserve magnitude
        out = self.gamma * (x * Nx) + self.beta + x             # (B, C, H, W)

        if self.training and torch.is_grad_enabled():
            assert not torch.isnan(out).any(), \
                f"NaN in {self.__class__.__name__} output"

        return out


class GRN(BaseFeatureCompetition):
    """
    Standard Global Response Normalization (ConvNeXt V2 baseline).

    Used when LE-GRN is disabled (use_le_grn=False).
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))    # (1, C, 1, 1)
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))     # (1, C, 1, 1)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply standard global response normalization (ConvNeXt V2 baseline).

        Residual is added inside.

        Args:
            x: (B, C, H, W) — input feature map.

        Returns:
            (B, C, H, W) — normalized feature map, residual already added.
        """
        B, C, H, W = x.shape                                   # (B, C, H, W)
        input_dtype = x.dtype

        # Global RMS per channel (float32 for numerical safety)
        Gx = torch.norm(
            x.float(), p=2, dim=(2, 3), keepdim=True
        )                                                       # (B, C, 1, 1)

        # Normalize by mean RMS across channels
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)    # (B, C, 1, 1)
        Nx = Nx.to(input_dtype)

        # Element-wise scaling + residual
        out = self.gamma * (x * Nx) + self.beta + x             # (B, C, H, W)

        if self.training and torch.is_grad_enabled():
            assert not torch.isnan(out).any(), \
                f"NaN in {self.__class__.__name__} output"

        return out


def build_feature_competition(
    dim: int,
    use_le_grn: bool = True,
    local_kernel_size: int = 3,
    eps: float = 1e-6,
) -> BaseFeatureCompetition:
    """Factory function for the feature competition module."""
    if use_le_grn:
        return LEGRN(dim=dim, local_kernel_size=local_kernel_size, eps=eps)
    else:
        return GRN(dim=dim, eps=eps)
