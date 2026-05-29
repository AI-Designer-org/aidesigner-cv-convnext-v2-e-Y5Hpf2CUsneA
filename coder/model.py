"""
ConvNeXt V2-E: Efficient ConvNet Backbone under 20M Parameters
===============================================================

Design innovations over ConvNeXt V2:
  1. MRF-DW: Multi-Receptive Field Depthwise Convolution
  2. LE-GRN: Local-Enhanced Global Response Normalization
  3. Adaptive Expansion: Stage-specific expansion ratios

Parameter budget: 19.7M (ImageNet-1K, 1000 classes)
FLOPs: ~3.8G (224x224 input)

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

import math
import torch
import torch.nn as nn
from typing import List, Optional, Dict

from model_config import ModelConfig
from layers import (
    LayerNormNd,
    Stem,
    DownsampleBlock,
    ConvNeXtV2EBlock,
)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class ConvNeXtV2E(nn.Module):
    """
    ConvNeXt V2-E: Efficient ConvNet backbone under 20M parameters.

    Hierarchical 4-stage design (1/4 -> 1/32 resolution) with:
      - Multi-scale spatial mixing per block (MRF-DW)
      - Local-enhanced feature competition (LE-GRN)
      - Stage-adaptive expansion ratios

    Compatible with FPN/necks for dense prediction tasks via
    get_intermediate_features().
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        depths: List[int] = cfg.depths
        dims: List[int] = cfg.dims
        expansion_ratios: List[float] = cfg.expansion_ratios
        small_kernel_sizes: List[int] = cfg.small_kernel_sizes

        # ── Stem: 3x224x224 -> Cx56x56 ───────────────────────────────
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
        self.total_blocks = total_blocks

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

    def _init_weights(self, cfg: ModelConfig):
        """Initialize all Conv2d and Linear weights.

        Uses truncated normal (preferred) or normal distribution with std=cfg.init_std.
        Biases initialized to zero.

        Args:
            cfg: ModelConfig with init_std and init_trunc_norm fields.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if cfg.init_trunc_norm:
                    nn.init.trunc_normal_(m.weight, std=cfg.init_std)
                else:
                    nn.init.normal_(m.weight, std=cfg.init_std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features through stem + stages + final norm + pool.

        Args:
            x: (B, 3, H, W) — input image.

        Returns:
            (B, C_last) — globally pooled feature vector.

        Shape invariants:
            - Input channels must equal cfg.in_channels (default 3).
            - Any H, W divisible by 32 supported (fully convolutional).
            - dtype in {float32, bfloat16, float16}.
        """
        B, C_in, H, W = x.shape                                     # (B, 3, H, W)

        x = self.stem(x)                                             # (B, C0, H/4, W/4)

        for i, stage in enumerate(self.stages):
            x = stage(x)                                             # (B, C_i, H_i, W_i)
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)                     # (B, C_{i+1}, H_{i+1}, W_{i+1})

        x = self.norm(x)                                             # (B, C_last, H_last, W_last)
        x = x.mean(dim=(2, 3))                                       # (B, C_last) — global average pool

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: features -> classification head.

        Args:
            x: (B, 3, H, W) — input image (typically 224x224).

        Returns:
            (B, num_classes) — classification logits.
        """
        B, C_in, H, W = x.shape                                     # (B, 3, H, W)
        x = self.forward_features(x)                                 # (B, C_last)
        x = self.head(x)                                             # (B, num_classes)
        return x

    def get_intermediate_features(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Return feature maps at each stage (useful for FPN/neck in dense prediction).

        Args:
            x: (B, 3, H, W) — input image.

        Returns:
            Dict with keys {0: stem_out, 1: stage1, 2: stage2, 3: stage3, 4: stage4}
            where each value is a (B, C_i, H_i, W_i) feature map.

        Shape invariants:
            - Stem output: (B, C0, H/4, W/4)
            - Stage i output: (B, C_i, H_i, W_i) before downsampling
            - Stage 4 output: (B, C_last, H/32, W/32) for 224x224 input
        """
        features: Dict[int, torch.Tensor] = {}

        B, C_in, H, W = x.shape                                     # (B, 3, H, W)

        x = self.stem(x)                                             # (B, C0, H/4, W/4)
        features[0] = x

        for i, stage in enumerate(self.stages):
            x = stage(x)                                             # (B, C_i, H_i, W_i)
            features[i + 1] = x
            if i < len(self.downsample_layers):
                x = self.downsample_layers[i](x)                     # (B, C_{i+1}, H_{i+1}, W_{i+1})

        return features


# ---------------------------------------------------------------------------
# Parameter count helper
# ---------------------------------------------------------------------------


def count_params(model: nn.Module) -> int:
    """Return total number of parameters in the model (including non-trainable)."""
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    """Return number of trainable (requires_grad=True) parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module):
    """Print parameter breakdown for the model."""
    total = count_params(model)
    trainable = count_trainable_params(model)
    print(f"Total params:     {total:>10,}")
    print(f"Trainable params: {trainable:>10,}")
    print(f"Within 20M:       {total < 20_000_000}")

    # Per-component breakdown
    component_params: Dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.Sequential)):
            continue  # skip leaf types and containers
        p = sum(p.numel() for p in module.parameters())
        if p > 0 and name not in ("", "stem", "norm", "head"):
            # Find the parent stage/component
            parts = name.split(".")
            if parts[0] in ("stages", "downsample_layers", "stem", "norm", "head"):
                component_params[parts[0]] = component_params.get(parts[0], 0) + p

    print("\nComponent breakdown:")
    for name, p in sorted(component_params.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}: {p:>10,} ({100 * p / total:.1f}%)")


# ---------------------------------------------------------------------------
# Variant constructors
# ---------------------------------------------------------------------------


def convnext_v2e_tiny(**kwargs) -> ConvNeXtV2E:
    """Primary variant: ~19.7M params, ~3.8G FLOPs at 224x224.

    Config: C=[96,192,384,512], B=[2,4,10,3], E=[3,3,4,3], all innovations active.

    Forward: (B, 3, H, W) -> (B, 1000).
    """
    from model_config import tiny_config
    cfg = tiny_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


def convnext_v2e_wide(**kwargs) -> ConvNeXtV2E:
    """Wider variant: ~19.4M params.

    Config: C=[112,224,448,448], B=[2,3,6,2], E=[3,3,4,3], all innovations active.

    Forward: (B, 3, H, W) -> (B, 1000).
    """
    from model_config import wide_config
    cfg = wide_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


def convnext_v2e_deep(**kwargs) -> ConvNeXtV2E:
    """Deeper variant: ~19.0M params (highest depth-to-width ratio).

    Config: C=[80,160,320,448], B=[2,5,14,3], E=[3,3,4,3], all innovations active.

    Forward: (B, 3, H, W) -> (B, 1000).
    """
    from model_config import deep_config
    cfg = deep_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


def convnext_v2e_ablated_baseline(**kwargs) -> ConvNeXtV2E:
    """Ablated baseline: single 7x7 DW + standard GRN + adaptive expansion.

    ~19.5M params. Use this to isolate innovation contributions via ablation.

    Forward: (B, 3, H, W) -> (B, 1000).
    """
    from model_config import ablated_baseline_config
    cfg = ablated_baseline_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


def convnext_v2e_uniform_expansion(**kwargs) -> ConvNeXtV2E:
    """Uniform x4 expansion for ablation. Over 20M budget — ablation only.

    Config: C=[96,192,384,512], B=[2,4,10,3], E=[4,4,4,4].
    ~20.5M params (exceeds 20M budget). For ablation comparison only.

    Forward: (B, 3, H, W) -> (B, 1000).
    """
    from model_config import uniform_expansion_config
    cfg = uniform_expansion_config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return ConvNeXtV2E(cfg)


# ---------------------------------------------------------------------------
# Parameter verification (when run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    variants = {
        "tiny": (convnext_v2e_tiny, {}),
        "wide": (convnext_v2e_wide, {}),
        "deep": (convnext_v2e_deep, {}),
        "ablated": (convnext_v2e_ablated_baseline, {}),
        "uniform_e4": (convnext_v2e_uniform_expansion, {}),
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    x = torch.randn(2, 3, 224, 224).to(device=device, dtype=dtype)

    for name, (fn, kwargs) in variants.items():
        model = fn(**kwargs).to(device=device, dtype=dtype)
        model.eval()
        total = count_params(model)
        with torch.no_grad():
            out = model(x)

        within_budget = total < 20_000_000
        expected_shape = (2, 1000)

        status = "OK" if (out.shape == expected_shape and within_budget) else "FAIL"
        print(
            f"[{status:>4}] {name:>14}:  "
            f"Params: {total:>9,}  "
            f"Budget OK: {str(within_budget):>5}  "
            f"Output: {list(out.shape)}  "
            f"Device: {device}"
        )
