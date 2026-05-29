"""
ConvNeXt V2-E: Model Configuration Dataclass
=============================================

Every hyperparameter appears here, never as a magic number.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    # ── Input ────────────────────────────────────────────────────────────
    in_channels: int = 3
    img_size: int = 224

    # ── Macro architecture ──────────────────────────────────────────────
    dims: List[int] = field(default_factory=lambda: [96, 192, 384, 512])
    depths: List[int] = field(default_factory=lambda: [2, 4, 10, 3])
    expansion_ratios: List[float] = field(
        default_factory=lambda: [3.0, 3.0, 4.0, 3.0]
    )

    # ── Innovation 1: MRF-DW ─────────────────────────────────────────────
    use_mrf_dw: bool = True
    # Small kernel size per stage
    small_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 5, 5])
    # Base kernel is always 7, matching ConvNeXt V2
    base_kernel_size: int = 7
    # Initial value for the sigmoid-gated mixing weight (0 → α=0.5)
    mrf_mix_init: float = 0.0

    # ── Innovation 2: LE-GRN ─────────────────────────────────────────────
    use_le_grn: bool = True
    le_grn_local_kernel: int = 3   # kernel size for local mean
    grn_eps: float = 1e-6

    # ── Stem ─────────────────────────────────────────────────────────────
    stem_kernel_size: int = 4
    stem_stride: int = 4

    # ── Downsampling ─────────────────────────────────────────────────────
    downsample_kernel_size: int = 2
    downsample_stride: int = 2

    # ── Regularization ───────────────────────────────────────────────────
    drop_path_rate: float = 0.1     # max stochastic depth rate
    label_smoothing: float = 0.1    # training-time only

    # ── Task head ────────────────────────────────────────────────────────
    num_classes: int = 1000

    # ── Numerics ─────────────────────────────────────────────────────────
    use_bias: bool = False           # ConvNeXt uses no bias in convs

    # ── Weight init ──────────────────────────────────────────────────────
    init_std: float = 0.02
    init_trunc_norm: bool = True

    # ── Gradient checkpointing ───────────────────────────────────────────
    use_checkpoint: bool = False

    # ── Internal (computed, not for user override) ───────────────────────
    total_blocks: int = 0

    def __post_init__(self):
        self.total_blocks = sum(self.depths)


@dataclass
class TrainingConfig:
    """Hyperparameters from the supervised training protocol."""
    optimizer: str = "AdamW"
    base_lr: float = 4e-3
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    scheduler: str = "cosine"
    warmup_epochs: int = 20
    total_epochs: int = 300
    batch_size: int = 4096
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    randaug_magnitude: int = 9
    randaug_num_ops: int = 2
    ema_decay: float = 0.9999
    drop_path_rate: float = 0.1


# ── Variant constructors ────────────────────────────────────────────────


def tiny_config() -> ModelConfig:
    """Primary variant: ~19.7M params, 3.8G FLOPs at 224x224.

    C=[96,192,384,512], B=[2,4,10,3], E=[3,3,4,3], all innovations active.
    """
    return ModelConfig(
        dims=[96, 192, 384, 512],
        depths=[2, 4, 10, 3],
        expansion_ratios=[3.0, 3.0, 4.0, 3.0],
        small_kernel_sizes=[3, 3, 5, 5],
    )


def wide_config() -> ModelConfig:
    """Wider, shallower: ~19.4M params.

    C=[112,224,448,448], B=[2,3,6,2], E=[3,3,4,3].
    """
    return ModelConfig(
        dims=[112, 224, 448, 448],
        depths=[2, 3, 6, 2],
        expansion_ratios=[3.0, 3.0, 4.0, 3.0],
        small_kernel_sizes=[3, 3, 5, 5],
    )


def deep_config() -> ModelConfig:
    """Narrower, deeper: ~19.0M params (highest depth-to-width ratio).

    C=[80,160,320,448], B=[2,5,14,3], E=[3,3,4,3].
    """
    return ModelConfig(
        dims=[80, 160, 320, 448],
        depths=[2, 5, 14, 3],
        expansion_ratios=[3.0, 3.0, 4.0, 3.0],
        small_kernel_sizes=[3, 3, 5, 5],
    )


def ablated_baseline_config() -> ModelConfig:
    """Ablated baseline: single 7x7 DW + standard GRN + adaptive expansion.

    ~19.5M params. Disables both MRF-DW and LE-GRN for ablation studies.
    """
    return ModelConfig(
        dims=[96, 192, 384, 512],
        depths=[2, 4, 10, 3],
        expansion_ratios=[3.0, 3.0, 4.0, 3.0],
        use_mrf_dw=False,
        use_le_grn=False,
    )


def uniform_expansion_config() -> ModelConfig:
    """Tests adaptive expansion: uniform x4 instead of [3,3,4,3].

    ~20.5M params (over 20M budget -- for ablation only).
    """
    return ModelConfig(
        dims=[96, 192, 384, 512],
        depths=[2, 4, 10, 3],
        expansion_ratios=[4.0, 4.0, 4.0, 4.0],
    )
