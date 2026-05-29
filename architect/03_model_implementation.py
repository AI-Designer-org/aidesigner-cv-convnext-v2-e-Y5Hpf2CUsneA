"""
ConvNeXt V2-E: Efficient ConvNet Backbone under 20M Parameters
================================================================

Design innovations:
  1. MRF-DW: Multi-Receptive Field Depthwise Convolution
  2. LE-GRN: Local-Enhanced Global Response Normalization
  3. Adaptive Expansion: Stage-specific expansion ratios

Parameter budget: 19.7M (ImageNet-1K, 1000 classes)
FLOPs: ~3.8G (224×224 input)

References:
  - ConvNeXt V2 (Woo et al., CVPR 2023)
  - ConvNeXt V1 (Liu et al., CVPR 2022)

Usage:
    from model_config import tiny_config
    cfg = tiny_config()
    model = ConvNeXtV2E(cfg)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


# ---------------------------------------------------------------------------
# Module definitions
# ---------------------------------------------------------------------------

class LayerNormNd(nn.Module):
    """Channel-first LayerNorm for (B, C, H, W) tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(dim=(1, 2, 3), keepdim=True)
        s = (x - u).pow(2).mean(dim=(1, 2, 3), keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class GRN(nn.Module):
    """
    Global Response Normalization (ConvNeXt V2 baseline).

    Gx = ||x||_2 over (H, W)     — global RMS per channel
    Nx = Gx / mean(Gx)           — normalize across channels
    out = γ · (x · Nx) + β + x  — element-wise scaling + residual
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class LEGRN(nn.Module):
    """
    Local-Enhanced Global Response Normalization (Innovation 2).

    Adds local mean subtraction before global normalization:
      x_local = x - μ_local(x)        — 3×3 avg pool
      Gx = ||x_local||_2              — global RMS of contrast-enhanced features
      Nx = Gx / mean(Gx)              — normalize across channels
      out = γ · (x · Nx) + β + x     — same output interface as GRN

    Zero additional parameters vs GRN.
    """

    def __init__(self, dim: int, local_kernel_size: int = 3, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.local_kernel_size = local_kernel_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Local mean subtraction (reflect padding to avoid edge artifacts)
        local_mean = F.avg_pool2d(
            x,
            kernel_size=self.local_kernel_size,
            stride=1,
            padding=self.local_kernel_size // 2,
            count_include_pad=False,
        )
        x_local = x - local_mean

        # Standard GRN on locally-enhanced features
        Gx = torch.norm(x_local, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * Nx) + self.beta + x


class MRFDWConv(nn.Module):
    """
    Multi-Receptive Field Depthwise Convolution (Innovation 1).

    Parallel depthwise convolutions (base_k × base_k ∥ small_k × small_k)
    mixed via learnable per-channel sigmoid-gated weights.

    α_ch = σ(w_ch)      — mixing weight per channel, initialized to 0.5
    out = α · DW_base(x) + (1-α) · DW_small(x)
    """

    def __init__(
        self,
        dim: int,
        base_kernel_size: int = 7,
        small_kernel_size: int = 3,
        mix_init: float = 0.0,
    ):
        super().__init__()

        # Base kernel (7×7, matching ConvNeXt V2)
        self.base_dw = nn.Conv2d(
            dim, dim,
            kernel_size=base_kernel_size,
            padding=base_kernel_size // 2,
            groups=dim,
            bias=False,
        )

        # Small kernel (3×3 in early stages, 5×5 in late stages)
        small_padding = small_kernel_size // 2
        self.small_dw = nn.Conv2d(
            dim, dim,
            kernel_size=small_kernel_size,
            padding=small_padding,
            groups=dim,
            bias=False,
        )

        # Per-channel mixing weight
        # Init to mix_init (default 0) → sigmoid(0) = 0.5 → equal weighting
        self.mix_weight = nn.Parameter(torch.full((1, dim, 1, 1), mix_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_dw(x)
        small_out = self.small_dw(x)
        alpha = torch.sigmoid(self.mix_weight)
        return alpha * base_out + (1.0 - alpha) * small_out


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
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ConvNeXtV2EBlock(nn.Module):
    """
    ConvNeXt V2-E block with MRF-DW and LE-GRN.

    Order:
      MRF-DW → LN → PW1 (expand) → GELU → PW2 (project) → LE-GRN → DropPath
      + residual
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
        if use_mrf_dw:
            self.dwconv = MRFDWConv(
                dim=dim,
                base_kernel_size=base_kernel_size,
                small_kernel_size=small_kernel_size,
                mix_init=mrf_mix_init,
            )
        else:
            self.dwconv = nn.Conv2d(
                dim, dim,
                kernel_size=base_kernel_size,
                padding=base_kernel_size // 2,
                groups=dim,
                bias=False,
            )

        # ── Channel mixing (inverted bottleneck) ──────────────────────
        self.norm = LayerNormNd(dim)
        self.pw1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=False)

        # ── Feature competition ───────────────────────────────────────
        if use_le_grn:
            self.grn = LEGRN(dim, local_kernel_size=le_grn_local_kernel, eps=grn_eps)
        else:
            self.grn = GRN(dim, eps=grn_eps)

        # ── Stochastic depth ──────────────────────────────────────────
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)      # spatial mixing
        x = self.norm(x)        # pre-norm
        x = self.pw1(x)         # expand
        x = self.act(x)         # activation
        x = self.pw2(x)         # project
        x = self.grn(x)         # feature competition
        x = self.drop_path(x)   # stochastic depth
        return shortcut + x


class DownsampleBlock(nn.Module):
    """
    ConvNeXt downsampling: LayerNorm → Conv2D 2×2 stride 2.
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
        return self.conv(self.norm(x))


class Stem(nn.Module):
    """
    ConvNeXt stem: 4×4 conv, stride 4 (no post-norm — matches V2 design).
    Maps 3×224×224 → C×56×56.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 96):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=4, stride=4,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ConvNeXtV2E(nn.Module):
    """
    ConvNeXt V2-E: Efficient ConvNet backbone under 20M parameters.

    Parameters
    ----------
    cfg : ModelConfig
        Full model configuration dataclass.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes

        # Per-stage parameters
        depths: List[int] = cfg.depths
        dims: List[int] = cfg.dims
        expansion_ratios: List[float] = cfg.expansion_ratios
        small_kernel_sizes: List[int] = cfg.small_kernel_sizes

        # ── Stem ──────────────────────────────────────────────────────
        self.stem = Stem(
            in_channels=cfg.in_channels,
            out_channels=dims[0],
        )

        # ── Stochastic depth schedule (linear per-block) ──────────────
        total_blocks = sum(depths)
        dpr = [
            x.item()
            for x in torch.linspace(0, cfg.drop_path_rate, total_blocks)
        ]

        # ── Build stages ──────────────────────────────────────────────
        self.stages = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        block_idx = 0
        for i, (depth, dim, exp_ratio, small_k) in enumerate(
            zip(depths, dims, expansion_ratios, small_kernel_sizes)
        ):
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(
                    ConvNeXtV2EBlock(
                        dim=dim,
                        expansion_ratio=exp_ratio,
                        use_mrf_dw=cfg.use_mrf_dw,
                        small_kernel_size=small_k,
                        mrf_mix_init=cfg.mrf_mix_init,
                        base_kernel_size=cfg.base_kernel_size,
                        use_le_grn=cfg.use_le_grn,
                        le_grn_local_kernel=cfg.le_grn_local_kernel,
                        grn_eps=cfg.grn_eps,
                        drop_path_prob=dpr[block_idx],
                    )
                )
                block_idx += 1

            self.stages.append(nn.Sequential(*stage_blocks))

            # Downsampling between stages (not after last)
            if i < len(depths) - 1:
                self.downsample_layers.append(
                    DownsampleBlock(dim, dims[i + 1])
                )

        # ── Head ──────────────────────────────────────────────────────
        self.norm = LayerNormNd(dims[-1])
        self.head = (
            nn.Linear(dims[-1], cfg.num_classes)
            if cfg.num_classes > 0
            else nn.Identity()
        )

        # ── Weight initialization ─────────────────────────────────────
        self._init_weights(cfg)

    def _init_weights(self, cfg):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if cfg.init_trunc_norm:
                    nn.init.trunc_normal_(m.weight, std=cfg.init_std)
                else:
                    nn.init.normal_(m.weight, std=cfg.init_std)
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

    def get_intermediate_features(self, x: torch.Tensor):
        """
        Return feature maps at each stage (useful for FPN/neck in dense prediction).

        Returns dict: {0: stem_out, 1: stage1, 2: stage2, 3: stage3, 4: stage4}
        """
        features = {}
        x = self.stem(x)
        features[0] = x
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features[i + 1] = x
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)
        return features


# ---------------------------------------------------------------------------
# Variant constructors
# ---------------------------------------------------------------------------


def convnext_v2e_tiny(**kwargs) -> ConvNeXtV2E:
    """
    Primary variant: ~19.7M params at 224×224.
    C=[96,192,384,512], B=[2,4,10,3], E=[3,3,4,3]
    """
    from model_config import tiny_config
    cfg = tiny_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


def convnext_v2e_wide(**kwargs) -> ConvNeXtV2E:
    """
    Wider variant: ~19.4M params.
    C=[112,224,448,448], B=[2,3,6,2], E=[3,3,4,3]
    """
    from model_config import wide_config
    cfg = wide_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


def convnext_v2e_deep(**kwargs) -> ConvNeXtV2E:
    """
    Deeper variant: ~19.0M params.
    C=[80,160,320,448], B=[2,5,14,3], E=[3,3,4,3]
    """
    from model_config import deep_config
    cfg = deep_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


def convnext_v2e_ablated_baseline(**kwargs) -> ConvNeXtV2E:
    """
    Ablated baseline: single 7×7 DW + standard GRN + adaptive expansion.
    ~19.5M params. Use this to isolate innovation contributions.
    """
    from model_config import ablated_baseline_config
    cfg = ablated_baseline_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


def convnext_v2e_uniform_expansion(**kwargs) -> ConvNeXtV2E:
    """
    Uniform ×4 expansion for ablation.
    ~20.5M params (hits budget limit — for ablation only).
    """
    from model_config import uniform_expansion_config
    cfg = uniform_expansion_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


# ---------------------------------------------------------------------------
# Parameter verification
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def count_params(model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    variants = {
        "tiny": (convnext_v2e_tiny, {}),
        "wide": (convnext_v2e_wide, {}),
        "deep": (convnext_v2e_deep, {}),
        "ablated": (convnext_v2e_ablated_baseline, {}),
        "uniform_e4": (convnext_v2e_uniform_expansion, {}),
    }

    x = torch.randn(2, 3, 224, 224)

    for name, (fn, kwargs) in variants.items():
        model = fn(**kwargs)
        total, trainable = count_params(model)
        out = model(x)
        within_budget = total < 20_000_000

        print(f"[{name:>14}]  "
              f"Params: {total:>9,}  "
              f"Trainable: {trainable:>9,}  "
              f"Within 20M: {within_budget}  "
              f"Output: {list(out.shape)}")
