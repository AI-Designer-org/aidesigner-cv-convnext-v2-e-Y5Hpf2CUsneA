# API Reference

---

## `model_config.py`

### `class ModelConfig`
Configuration dataclass for ConvNeXt V2-E. Every hyperparameter appears here — no magic numbers in implementation code.

**Fields:**
- `in_channels: int` (default=3) — Number of input image channels (RGB)
- `img_size: int` (default=224) — Input image spatial size for training
- `dims: List[int]` (default=[96, 192, 384, 512]) — Channel dimensions per stage (4 stages)
- `depths: List[int]` (default=[2, 4, 10, 3]) — Number of blocks per stage
- `expansion_ratios: List[float]` (default=[3.0, 3.0, 4.0, 3.0]) — MLP expansion ratio per stage
- `use_mrf_dw: bool` (default=True) — Enable Multi-Receptive Field Depthwise Convolution
- `small_kernel_sizes: List[int]` (default=[3, 3, 5, 5]) — Small kernel size for MRF-DW per stage
- `base_kernel_size: int` (default=7) — Base kernel size for MRF-DW (always 7, matching ConvNeXt V2)
- `mrf_mix_init: float` (default=0.0) — Initial value for MRF-DW mixing weight; sigmoid(0) = 0.5
- `use_le_grn: bool` (default=True) — Enable Local-Enhanced GRN
- `le_grn_local_kernel: int` (default=3) — Kernel size for local mean subtraction in LE-GRN
- `grn_eps: float` (default=1e-6) — Epsilon for GRN/LE-GRN numerical stability
- `stem_kernel_size: int` (default=4) — Stem convolution kernel size
- `stem_stride: int` (default=4) — Stem convolution stride
- `downsample_kernel_size: int` (default=2) — Downsampling convolution kernel size
- `downsample_stride: int` (default=2) — Downsampling convolution stride
- `drop_path_rate: float` (default=0.1) — Maximum stochastic depth rate (linear schedule)
- `dropout: float` (default=0.0) — Dropout rate (not used in ConvNeXt blocks)
- `label_smoothing: float` (default=0.1) — Label smoothing for training
- `num_classes: int` (default=1000) — Number of output classes
- `use_bias: bool` (default=False) — Whether convolutions use bias (ConvNeXt convention: no bias)
- `init_std: float` (default=0.02) — Standard deviation for weight initialization
- `init_trunc_norm: bool` (default=True) — Use truncated normal (vs regular normal) for init
- `use_checkpoint: bool` (default=False) — Enable gradient checkpointing
- `total_blocks: int` (computed) — Sum of depths, auto-calculated in `__post_init__`

### `class TrainingConfig`
Hyperparameters for the supervised training protocol.

**Fields:**
- `optimizer: str` (default="AdamW")
- `base_lr: float` (default=4e-3)
- `weight_decay: float` (default=0.05)
- `beta1: float` (default=0.9)
- `beta2: float` (default=0.999)
- `scheduler: str` (default="cosine")
- `warmup_epochs: int` (default=20)
- `total_epochs: int` (default=300)
- `batch_size: int` (default=4096)
- `label_smoothing: float` (default=0.1)
- `mixup_alpha: float` (default=0.8)
- `cutmix_alpha: float` (default=1.0)
- `randaug_magnitude: int` (default=9)
- `randaug_num_ops: int` (default=2)
- `ema_decay: float` (default=0.9999)
- `drop_path_rate: float` (default=0.1)

### Variant constructors

- `tiny_config() -> ModelConfig` — Primary variant: C=[96,192,384,512], B=[2,4,10,3], E=[3,3,4,3]; ~19.7M params
- `wide_config() -> ModelConfig` — Wider variant: C=[112,224,448,448], B=[2,3,6,2]; ~19.4M params
- `deep_config() -> ModelConfig` — Deeper variant: C=[80,160,320,448], B=[2,5,14,3]; ~19.0M params
- `ablated_baseline_config() -> ModelConfig` — Single 7×7 DW + standard GRN + adaptive expansion; ~19.5M params
- `uniform_expansion_config() -> ModelConfig` — Uniform ×4 expansion for ablation; ~20.5M params (over budget)

---

## `mrf_dw.py`

### `class BaseSpatialOperator(ABC, nn.Module)`
Abstract base class for the spatial mixing operator.

**Methods:**
- `forward(x: Tensor) -> Tensor` — Applies spatial mixing. `x`: `(B, C, H, W)`. Returns: `(B, C, H, W)`.

### `class MRFDWConv(BaseSpatialOperator)`
Multi-Receptive Field Depthwise Convolution (Innovation 1).

Runs two parallel depthwise convolutions (base kernel + small kernel) and blends them with a learnable per-channel sigmoid-gated weight.

**Constructor:** `MRFDWConv(dim, base_kernel_size=7, small_kernel_size=3, mix_init=0.0)`
- `dim`: Number of channels
- `base_kernel_size`: Kernel size for base DW conv (default 7)
- `small_kernel_size`: Kernel size for small DW conv (default 3)
- `mix_init`: Initial value for mixing weight; sigmoid(0)=0.5

**Forward:** `(B, C, H, W) → (B, C, H, W)` — learned per-channel convex combination of two depthwise outputs.

**Shape invariants:**
- Input and output have identical shape
- `C` must match `dim` passed to constructor
- `H, W` must be ≥ max(base_kernel_size, small_kernel_size)

### `class StandardDWConv(BaseSpatialOperator)`
Standard single-kernel depthwise convolution (ConvNeXt V2 baseline).

Used when `use_mrf_dw=False`. Provides the same interface as `MRFDWConv`.

**Constructor:** `StandardDWConv(dim, kernel_size=7)`

**Forward:** `(B, C, H, W) → (B, C, H, W)`

### `build_spatial_mixer(dim, use_mrf_dw, base_kernel_size, small_kernel_size, mix_init) -> BaseSpatialOperator`
Factory function. Returns `MRFDWConv` if `use_mrf_dw=True`, else `StandardDWConv`.

---

## `le_grn.py`

### `class BaseFeatureCompetition(ABC, nn.Module)`
Abstract base class for feature competition/normalization.

**Methods:**
- `forward(x: Tensor) -> Tensor` — Applies channel normalization. `x`: `(B, C, H, W)`. Returns: `(B, C, H, W)`.

### `class LEGRN(BaseFeatureCompetition)`
Local-Enhanced Global Response Normalization (Innovation 2).

Augments GRN with local mean subtraction before global RMS normalization. Zero additional parameters vs GRN.

**Constructor:** `LEGRN(dim, local_kernel_size=3, eps=1e-6)`
- `dim`: Number of channels
- `local_kernel_size`: Kernel size for avg-pool local mean (default 3)
- `eps`: Numerical stability epsilon

**Forward:** `(B, C, H, W) → (B, C, H, W)`

Computation:
1. `local_mean = AvgPool2D(x, kernel_size=k)` — local mean
2. `x_local = x - local_mean` — contrast enhancement
3. `Gx = ||x_local||₂` over `(H, W)` — global RMS
4. `Nx = Gx / mean(Gx) + eps` — channel normalization
5. `out = γ · (x · Nx) + β + x` — scaling + residual

**Shape invariants:**
- Input and output identical shape
- L2 norm computation cast to float32 for numerical safety

### `class GRN(BaseFeatureCompetition)`
Standard Global Response Normalization (ConvNeXt V2 baseline).

Used when `use_le_grn=False`. Same parameter count as LEGRN.

**Constructor:** `GRN(dim, eps=1e-6)`

**Forward:** `(B, C, H, W) → (B, C, H, W)`

### `build_feature_competition(dim, use_le_grn, local_kernel_size, eps) -> BaseFeatureCompetition`
Factory function. Returns `LEGRN` if `use_le_grn=True`, else `GRN`.

---

## `layers.py`

### `class LayerNormNd(nn.Module)`
Channel-first LayerNorm for `(B, C, H, W)` tensors.

Compatible with `nn.LayerNorm` but operates on conv-net format. Casts mean/variance computation to float32 for low-precision safety.

**Constructor:** `LayerNormNd(normalized_shape, eps=1e-6)`

**Forward:** `(B, C, H, W) → (B, C, H, W)` — normalized, with learnable affine transform.

**Shape invariants:**
- `C` must equal `normalized_shape`
- Mean/variance computed in float32 regardless of input dtype
- dtype restriction: fp16 and bf16 supported via fp32 cast

### `class DropPath(nn.Module)`
Stochastic depth for residual blocks. Drops entire samples during training.

**Constructor:** `DropPath(drop_prob=0.0)`

**Forward:** `(B, C, H, W) → (B, C, H, W)` — identity in eval; drops samples in train.

**Shape invariants:**
- Input and output identical shape
- Drop is sample-wise (broadcast across all non-batch dims)

### `class Stem(nn.Module)`
ConvNeXt stem: 4×4 convolution, stride 4. Maps 3×224×224 → C×56×56.

**Constructor:** `Stem(in_channels=3, out_channels=96)`

**Forward:** `(B, C_in, H, W) → (B, C_out, H/4, W/4)`

### `class DownsampleBlock(nn.Module)`
ConvNeXt downsampling: LayerNorm → Conv2D 2×2, stride 2.

**Constructor:** `DownsampleBlock(in_channels, out_channels)`

**Forward:** `(B, C_in, H, W) → (B, C_out, H/2, W/2)`

### `class ConvNeXtV2EBlock(nn.Module)`
The core building block of ConvNeXt V2-E.

Architecture: MRF-DW (or standard 7×7 DW) → LayerNorm → Conv1×1 expand → GELU → Conv1×1 project → LE-GRN (or standard GRN) → DropPath + residual.

**Constructor:**
```python
ConvNeXtV2EBlock(
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
)
```

**Methods:**
- `forward(x, use_checkpoint=False) -> Tensor` — Block forward pass. `x`: `(B, C, H, W)`. Returns `(B, C, H, W)`. If `use_checkpoint=True` and `self.training`, uses gradient checkpointing.
- `_forward(x) -> Tensor` — Internal forward without checkpoint wrapping.

**Shape invariants:**
- Input and output identical shape `(B, C, H, W)`
- Internal expansion: `(B, C, H, W) → (B, e·C, H, W) → (B, C, H, W)`
- Residual is NOT added inside the feature competition module; it is added in this block after DropPath

---

## `model.py`

### `class ConvNeXtV2E(nn.Module)`
The full ConvNeXt V2-E backbone.

Hierarchical 4-stage design (1/4 → 1/32 resolution) with:
- Multi-scale spatial mixing per block (MRF-DW)
- Local-enhanced feature competition (LE-GRN)
- Stage-adaptive expansion ratios

Compatible with FPN/necks for dense prediction tasks via `get_intermediate_features()`.

**Constructor:** `ConvNeXtV2E(cfg: ModelConfig)`

**Methods:**

- `forward(x: Tensor) -> Tensor` — Full forward pass. `x`: `(B, 3, H, W)`. Returns `(B, num_classes)` — classification logits.

- `forward_features(x: Tensor) -> Tensor` — Extract features through stem + stages + final norm + pool. `x`: `(B, 3, H, W)`. Returns `(B, C_last)` — globally pooled features.

- `get_intermediate_features(x: Tensor) -> Dict[int, Tensor]` — Return feature maps at each stage. Returns: `{0: stem_out, 1: stage1, 2: stage2, 3: stage3, 4: stage4}`. Useful for FPN/necks in detection/segmentation.

- `_init_weights(cfg)` — Initialize all Conv2d and Linear weights (truncated normal or normal, based on config).

**Shape invariants:**
- Input: `(B, C_in, H, W)` with `C_in = cfg.in_channels` (default 3)
- Intermediate: stage 1 → `(B, C_0, H/4, W/4)`, stage 2 → `(B, C_1, H/8, W/8)`, stage 3 → `(B, C_2, H/16, W/16)`, stage 4 → `(B, C_3, H/32, W/32)`
- Output: `(B, cfg.num_classes)`
- dtype: float32, bfloat16, float16 (LayerNorm and GRN cast internally to float32 for reductions)

### Variant constructors

- `convnext_v2e_tiny(**kwargs) -> ConvNeXtV2E` — Primary variant: ~19.7M params. C=[96,192,384,512], B=[2,4,10,3], E=[3,3,4,3]
- `convnext_v2e_wide(**kwargs) -> ConvNeXtV2E` — Wider variant: ~19.4M params. C=[112,224,448,448], B=[2,3,6,2]
- `convnext_v2e_deep(**kwargs) -> ConvNeXtV2E` — Deeper variant: ~19.0M params. C=[80,160,320,448], B=[2,5,14,3]
- `convnext_v2e_ablated_baseline(**kwargs) -> ConvNeXtV2E` — Ablated baseline: single 7×7 DW + standard GRN. ~19.5M params.
- `convnext_v2e_uniform_expansion(**kwargs) -> ConvNeXtV2E` — Uniform ×4 expansion (over budget). For ablation only.

### Helper functions

- `count_params(model: nn.Module) -> int` — Total number of parameters
- `count_trainable_params(model: nn.Module) -> int` — Number of trainable parameters
- `print_model_summary(model: nn.Module)` — Print parameter breakdown by component

---

## `model_config.py` (variant configs — already in API above)
