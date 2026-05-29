# Architecture

## 1. Motivation

ConvNeXt V2 (Woo et al., CVPR 2023) established that modernized ConvNets with depthwise convolutions, inverted bottlenecks, and Global Response Normalization can compete with Vision Transformers at scale. However, the architecture was designed for the 28M–200M parameter range, and several design choices become suboptimal below 20M parameters.

**Three specific gaps** drive the ConvNeXt V2-E design:

1. **Uniform 7×7 depthwise convolution across all stages.** Every ConvNeXt V2 block uses the same kernel size, regardless of spatial resolution. Early stages (56×56) process fine details and would benefit from smaller kernels, while later stages (7×7) need broader context. Multi-scale kernel mixing within each block — conceptually related to Inception (Szegedy et al., 2015) and MixConv (Tan & Le, 2019) — is absent from the ConvNeXt family.

2. **Global Response Normalization discards local contrast.** GRN computes RMS over all H×W spatial positions, then normalizes across channels. This ignores local intensity differences that carry texture and edge information. For narrow models (C ≤ 512), where each channel encodes more distinct features, local contrast provides useful signal at no additional parameter cost.

3. **Uniform ×4 expansion inherited from Swin Transformer.** The 3:3:9:3 depth distribution and uniform expansion ratio 4 were designed for Swin-T at 28M parameters. At smaller scales, the optimal allocation of channel capacity and expansion across stages differs — early stages are FLOPs-dominated (high spatial resolution), while middle stages carry most semantic processing.

**Hypothesis tested by this architecture**: A ConvNeXt V2 variant under 20M parameters can match or exceed the representational capacity of ConvNeXt V2-T (28M) per parameter by combining (a) multi-scale depthwise mixing with learnable per-channel fusion, (b) local-enhanced normalization at zero parameter cost, and (c) stage-adaptive expansion ratios that allocate capacity where it matters most.

---

## 2. At a glance

```
                     Input Image (B, 3, 224, 224)
                                │
                                ▼
                     ┌─────────────────────┐
                     │       Stem          │  Conv2D 4×4, stride 4
                     │  (3 → 96 ch)        │
                     └─────────┬───────────┘
                               │  (B, 96, 56, 56)
                               ▼
          ┌─────────────────────────────────────────┐
          │           Stage 1 (×2 blocks)           │
          │  ┌──────────────────────────────────┐   │
          │  │  ConvNeXtV2EBlock(C=96, e=3)     │   │
          │  │  ┌───────────────────────────┐   │   │
          │  │  │  MRF-DW (7×7 ∥ 3×3)      │   │   │
          │  │  │  LayerNorm                │   │   │
          │  │  │  Conv1×1 (96→288) [GELU]  │   │   │
          │  │  │  Conv1×1 (288→96)         │   │   │
          │  │  │  LE-GRN                   │   │   │
          │  │  │  DropPath + residual      │   │   │
          │  │  └───────────────────────────┘   │   │
          │  └──────────────────────────────────┘   │
          └─────────┬───────────────────────────────┘
                    │  (B, 96, 56, 56)
                    ▼
          ┌─────────────────────┐
          │  Downsample 1       │  LN → Conv2D 2×2 stride 2
          │  (96 → 192 ch)      │
          └─────────┬───────────┘
                    │  (B, 192, 28, 28)
                    ▼
          ┌─────────────────────────────────────────┐
          │           Stage 2 (×4 blocks)           │
          │  ┌──────────────────────────────────┐   │
          │  │  ConvNeXtV2EBlock(C=192, e=3)    │   │
          │  │  ┌───────────────────────────┐   │   │
          │  │  │  MRF-DW (7×7 ∥ 3×3)      │   │   │
          │  │  │  LayerNorm                │   │   │
          │  │  │  Conv1×1 (192→576) [GELU] │   │   │
          │  │  │  Conv1×1 (576→192)        │   │   │
          │  │  │  LE-GRN                   │   │   │
          │  │  │  DropPath + residual      │   │   │
          │  │  └───────────────────────────┘   │   │
          │  └──────────────────────────────────┘   │
          └─────────┬───────────────────────────────┘
                    │  (B, 192, 28, 28)
                    ▼
          ┌─────────────────────┐
          │  Downsample 2       │  LN → Conv2D 2×2 stride 2
          │  (192 → 384 ch)     │
          └─────────┬───────────┘
                    │  (B, 384, 14, 14)
                    ▼
          ┌─────────────────────────────────────────┐
          │          Stage 3 (×10 blocks)           │  ◄── Bulk of capacity
          │  ┌──────────────────────────────────┐   │      10 blocks × e=4
          │  │  ConvNeXtV2EBlock(C=384, e=4)    │   │
          │  │  ┌───────────────────────────┐   │   │
          │  │  │  MRF-DW (7×7 ∥ 5×5)      │   │   │  ◄── Larger small kernel
          │  │  │  LayerNorm                │   │   │      at lower resolution
          │  │  │  Conv1×1 (384→1536) [GELU]│   │   │
          │  │  │  Conv1×1 (1536→384)       │   │   │
          │  │  │  LE-GRN                   │   │   │
          │  │  │  DropPath + residual      │   │   │
          │  │  └───────────────────────────┘   │   │
          │  └──────────────────────────────────┘   │  × 10
          └─────────┬───────────────────────────────┘
                    │  (B, 384, 14, 14)
                    ▼
          ┌─────────────────────┐
          │  Downsample 3       │  LN → Conv2D 2×2 stride 2
          │  (384 → 512 ch)     │
          └─────────┬───────────┘
                    │  (B, 512, 7, 7)
                    ▼
          ┌─────────────────────────────────────────┐
          │           Stage 4 (×3 blocks)           │
          │  ┌──────────────────────────────────┐   │
          │  │  ConvNeXtV2EBlock(C=512, e=3)    │   │
          │  │  ┌───────────────────────────┐   │   │
          │  │  │  MRF-DW (7×7 ∥ 5×5)      │   │   │
          │  │  │  LayerNorm                │   │   │
          │  │  │  Conv1×1 (512→1536) [GELU]│   │   │
          │  │  │  Conv1×1 (1536→512)       │   │   │
          │  │  │  LE-GRN                   │   │   │
          │  │  │  DropPath + residual      │   │   │
          │  └───────────────────────────┘   │   │
          │  └──────────────────────────────────┘   │
          └─────────┬───────────────────────────────┘
                    │  (B, 512, 7, 7)
                    ▼
          ┌─────────────────────┐
          │  LayerNorm          │  (B, 512, 7, 7)
          │  Global Avg Pool    │  (B, 512)
          │  Linear(512, 1000)  │  (B, 1000)
          └─────────┬───────────┘
                    │
                    ▼
              Output Logits
```

| Property | Value |
|---|---|
| Parameter count (default tiny config) | 19,670,632 |
| Time complexity | O(H·W·C²) per block (dominated by 1×1 convs); O(H·W·C·k²) for depthwise |
| Space complexity | O(B·C·H·W) activations; O(C²) weights per block |
| Hardware requirements | Any GPU with ≥4GB VRAM; fused DW conv kernels beneficial (Torch compile, TensorRT) |
| Custom kernels required | None — standard PyTorch ops only |

---

## 3. The core component: ConvNeXtV2EBlock

### 3.1 Intuition

The ConvNeXt V2-E block follows the same inverted-bottleneck structure as ConvNeXt V2: spatial mixing via depthwise convolution, followed by channel expansion, nonlinearity, projection, and feature competition. Two modifications change the behavior:

**MRF-DW** replaces the single 7×7 depthwise conv with a parallel pair (7×7 + smaller kernel). A per-channel sigmoid-gated weight blends the two outputs, letting each channel decide how much local vs. medium-range spatial context to use. Channels processing fine textures can weight the small kernel more; channels processing broader patterns can weight the large kernel. The mixing weight starts at 0.5 (equal blend) and is learned end-to-end.

**LE-GRN** augments the Global Response Normalization step with a local mean subtraction. Before computing the per-channel RMS (which will be normalized across channels), it subtracts a 3×3 average-pooled version of the feature map. This `x - local_mean(x)` operation emphasizes local contrast — edges, texture boundaries, and high-frequency details — without adding any learned parameters beyond GRN's existing γ and β.

### 3.2 Equations

**Per-block parameter formula** (for a block with dimension `C` and expansion ratio `e`):

```
P_block(C, e) = 49·C   (DW 7×7 base)
              + 2·C    (LayerNorm γ, β)
              + e·C²   (PW1 expand: C → e·C)
              + e·C²   (PW2 project: e·C → C)
              + 2·C    (GRN γ, β)
              = 2·e·C² + 53·C
```

**MRF-DW overhead** (with small kernel `k_s`):

```
Δ_MRF(C, k_s) = C·(k_s² + 1)
```

This adds a second depthwise conv (C·k_s² weights) plus the mixing weight (C values), replacing a single 49·C DW with (49·C + k_s²·C + C). The sigmoid-gated mixing is:

```
α       = σ(w)               where w ∈ ℝ^C initialized to 0
output  = α ⊙ DW_7(x) + (1−α) ⊙ DW_{k_s}(x)
```

**LE-GRN computation** (zero additional parameters vs GRN):

```
μ_local(x)  = AvgPool2D(x, kernel=3, stride=1, padding=1)
x_local     = x − μ_local(x)
Gx          = ||x_local||₂ over (H, W)       [global RMS of contrast-enhanced features]
Nx          = Gx / (mean_ch(Gx) + ε)         [normalize across channels]
output      = γ · (x · Nx) + β + x           [γ, β ∈ ℝ^C, initialized to 0]
```

Note that the final scaling `γ · (x · Nx)` uses the original `x`, not `x_local`. This preserves the overall activation magnitude while modulating it by the channel-competition factor derived from locally-enhanced features.

**Stage total** (with `B` blocks of dimension `C`, expansion `e`, MRF-DW small kernel `k_s`):

```
P_stage = B · [2·e·C² + 53·C + C·(k_s² + 1)]
```

### 3.3 Reference implementation walk-through

From `layers.py`, the `ConvNeXtV2EBlock._forward` method (40 lines shown):

```python
def _forward(self, x: torch.Tensor) -> torch.Tensor:
    shortcut = x                                  # (B, C, H, W) — save for residual

    # ── Spatial mixing: MRF-DW or standard 7×7 DW ─────────────────
    x = self.spatial_mixer(x)                     # (B, C, H, W) — learned multi-scale blend

    # ── Pre-norm ───────────────────────────────────────────────────
    x = self.norm(x)                              # (B, C, H, W) — LayerNorm (fp32-safe)

    # ── Inverted bottleneck ────────────────────────────────────────
    x = self.pw1(x)                               # (B, C, H, W) → (B, e·C, H, W) — expand
    x = self.act(x)                               # (B, e·C, H, W) — GELU non-linearity
    x = self.pw2(x)                               # (B, e·C, H, W) → (B, C, H, W) — project

    # ── Feature competition: LE-GRN or standard GRN ────────────────
    x = self.feature_competition(x)               # (B, C, H, W) — channel normalization

    # ── Stochastic depth ───────────────────────────────────────────
    x = self.drop_path(x)                         # (B, C, H, W) — random dropping

    return shortcut + x                           # (B, C, H, W) — residual connection
```

Shape invariants at each step: input and output are always `(B, C, H, W)`. The only shape change inside the block is the expansion in the bottleneck (`C → e·C → C`). Spatial dimensions are preserved throughout (all convolutions use padding).

---

## 4. Tensor shape evolution

For the default tiny config (224×224 input, batch=2):

| Stage | Operation | Shape | Notes |
|---|---|---|---|
| Input | Image | (2, 3, 224, 224) | RGB, float32 |
| Stem | Conv2D 4×4, stride 4 | (2, 96, 56, 56) | 1/4 spatial resolution |
| Stage 1 | 2× ConvNeXtV2EBlock(C=96, e=3) | (2, 96, 56, 56) | MRF-DW: 7×7 ∥ 3×3 |
| Downsample 1 | LN → Conv2D 2×2, stride 2 | (2, 192, 28, 28) | 1/8 spatial resolution |
| Stage 2 | 4× ConvNeXtV2EBlock(C=192, e=3) | (2, 192, 28, 28) | MRF-DW: 7×7 ∥ 3×3 |
| Downsample 2 | LN → Conv2D 2×2, stride 2 | (2, 384, 14, 14) | 1/16 spatial resolution |
| Stage 3 | 10× ConvNeXtV2EBlock(C=384, e=4) | (2, 384, 14, 14) | MRF-DW: 7×7 ∥ 5×5; bulk of capacity |
| Downsample 3 | LN → Conv2D 2×2, stride 2 | (2, 512, 7, 7) | 1/32 spatial resolution |
| Stage 4 | 3× ConvNeXtV2EBlock(C=512, e=3) | (2, 512, 7, 7) | MRF-DW: 7×7 ∥ 5×5 |
| Norm | LayerNorm | (2, 512, 7, 7) | Channel-first LN |
| Pool | Global avg pool | (2, 512) | mean(dim=(2,3)) |
| Head | Linear(512, 1000) | (2, 1000) | Classification logits |

---

## 5. Design decisions

| Decision | Alternative considered | Why we chose this | Trade-off accepted |
|---|---|---|---|
| **MRF-DW** — parallel 7×7 + smaller kernel with sigmoid-gated per-channel mixing | Single 7×7 DW (V2 baseline); dilated 7×7; single learned kernel size per channel | Multi-scale features at <0.8% param overhead; per-channel mixing lets each channel specialize | 4–8% extra FLOPs from second depthwise branch; DW convs are memory-bound, so wall-clock impact exceeds the FLOPs fraction for small batches |
| **LE-GRN** — local mean subtraction before global RMS | Standard GRN; InstanceNorm; LayerNorm | Zero parameter overhead; preserves GRN's residual interface; local contrast emphasizes texture boundaries without additional learned parameters | Fixed 3×3 local window may miss larger-scale contrast; reflection-padded avg pool adds minor compute |
| **Adaptive expansion** — [3,3,4,3] per stage | Uniform ×4 (V2 default); uniform ×3; ×3.5 in stage 3 | Saves ~1.9M params vs uniform ×4 (verified by smoke test); allocates ×4 only to stage 3 (10 blocks, where most semantic processing occurs) | Stage 3 caps at ×4 — further expansion might help but would exceed budget |
| **Depth distribution** — [2,4,10,3] blocks per stage | [3,3,9,3] (V2-T layout) | More blocks in stage 3 (10 vs 9) where spatial size is moderate (14×14) and FLOPs-per-block is efficient; fewer in early stages where spatial size is large | Stage 3 contains 61.5% of parameters — any degradation there has outsized impact |
| **Pre-norm** (LN after DW, before pointwise expansion) | Post-norm (LN after PW2, before residual) | Stabilizes activations entering the wide expansion layer; matches ConvNeXt V2 convention | Norm in the forward path adds a small compute cost at each block |
| **Stem: Conv2D 4×4 stride 4, no post-norm** | 4×4 conv + LayerNorm (V1 style); two 3×3 convs (ResNet-style) | Matches ConvNeXt V2 (norm is in first block); single conv minimizes early downsampling params (4.6K total) | No explicit norm after stem — first block's LN must handle potentially large activations |
| **Capped stage 4 at 512 channels** (vs V2-T's 768) | 640 channels for more capacity | 512 at e=3 stays within budget (23.9% of total params); 640 at e=4 would cost ~6.1M more, exceeding 20M | Less representational capacity in final stage — may hurt fine-grained discrimination |
| **Sigmoid-gated mixing for MRF-DW** | Softmax over N kernels; learnable scalar per stage | Ensures convex combination α∈(0,1); gradient flows through both branches at init | Sigmoid can saturate if weights grow large — monitored via alpha distribution during training |
| **LayerNorm with fp32 cast** | Compute LN in input dtype (bf16/fp16) | Prevents NaN in norm reduction for low-precision training (critical for bf16) | Minor overhead from dtype cast on each forward pass |

---

## 6. Domain-specific considerations (CV)

| Concern | How ConvNeXt V2-E addresses it |
|---|---|
| **Spatial handling at different resolutions** | Fully convolutional 4-stage pyramid (1/4 → 1/32). Global avg pool at head removes resolution dependence. Tested at 128×128 through 384×384. |
| **Multi-scale features** | Hierarchical pyramid stages. MRF-DW adds within-block multi-scale. Compatible with FPN/necks via `get_intermediate_features()`. |
| **Translation equivariance** | Depthwise convolutions are translation-equivariant. Small shifts produce similar logits (verified: cosine similarity > 0.5 for 8px shift on structured inputs). |
| **Inductive bias vs. data hunger** | Strong locality bias means faster convergence than ViTs at ImageNet-1K scale. No need for large-scale pretraining or extensive data augmentation beyond standard recipe. |
| **Normalization choice** | LayerNorm (not BatchNorm) avoids batch-size dependence. Works with any batch size including 1. Compatible with gradient checkpointing. |
| **Stochastic depth** | Linear DropPath schedule (0 → 0.1) across 19 blocks. Early blocks train fully; deeper blocks get stronger regularization. |

---

## 7. Known limitations

- **All accuracy claims are `TODO: unverified`.** No ImageNet-1K training has been performed. The expected accuracy range (80.5–81.5% supervised, 224×224) is extrapolated from the ConvNeXt V2 scaling curve. See [BENCHMARKS.md](BENCHMARKS.md) for the current verification status.
- **MRF-DW FLOPs overhead may not translate to wall-clock time savings.** Depthwise convolutions are memory-bandwidth-bound. The 4–8% FLOPs increase from the second DW branch may cause a larger wall-clock slowdown than the FLOPs fraction suggests, especially at small batch sizes where memory latency dominates.
- **FCMAE compatibility is untested.** LE-GRN's local mean subtraction may interact poorly with the zeroed-out regions in masked autoencoding. A dedicated FCMAE pretraining experiment is needed to verify compatibility.
- **Stage 3 parameter dominance (61.5% of total).** If Stage 3 underperforms (e.g., gradient vanishing in deep blocks), a disproportionate fraction of capacity is wasted. The DropPath schedule and gradient checkpointing mitigate but do not eliminate this risk.
- **No dense prediction evaluation.** The architecture is designed for backbone use in detection/segmentation via `get_intermediate_features()`, but no COCO or ADE20K experiments exist.
- **Sigmoid gate saturation risk.** If the mixing weights α saturate to 0 or 1 for all channels, the second kernel provides no benefit. The initialization (α=0.5) and per-channel learning reduce this risk but do not guarantee diverse mixing.
- **No comparison against EfficientNet-B2/B3, Swin-T, or RegNetY baselines.** These comparisons require external model weights and evaluation infrastructure not currently set up.
