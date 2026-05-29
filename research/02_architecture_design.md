# ConvNeXt V2-E: Architecture Specification

## Overview

**ConvNeXt V2-E** (Efficient) is a redesigned variant of ConvNeXt V2 targeting <20M parameters while preserving or improving representational capacity through three architectural innovations:

1. **MRF-DW**: Multi-Receptive Field Depthwise Convolution
2. **LE-GRN**: Local-Enhanced Global Response Normalization
3. **Adaptive Expansion**: Stage-specific expansion ratios

---

## 1. Base Configuration

### Macro Architecture

```
Input (3×224×224)
    │
    ├── Stem: Conv2D 4×4, stride 4 → 96 channels (1/4 scale)
    │
    ├── Stage 1 (56×56, C=96, 2 blocks, e=3)
    │   └── Downsample: LN → Conv2D 2×2, stride 2
    │
    ├── Stage 2 (28×28, C=192, 4 blocks, e=3)
    │   └── Downsample: LN → Conv2D 2×2, stride 2
    │
    ├── Stage 3 (14×14, C=384, 10 blocks, e=4)
    │   └── Downsample: LN → Conv2D 2×2, stride 2
    │
    ├── Stage 4 (7×7, C=512, 3 blocks, e=3)
    │
    └── Head: LayerNorm → Global Avg Pool → Linear (N-class)
```

### Configuration Summary

| Parameter | ConvNeXt V2-T | ConvNeXt V2-E (Proposed) |
|---|---|---|
| Channel pattern | [96, 192, 384, 768] | [96, 192, 384, 512] |
| Block counts | [3, 3, 9, 3] | [2, 4, 10, 3] |
| Expansion ratios | [4, 4, 4, 4] | [3, 3, 4, 3] |
| Total blocks | 18 | 19 |
| Stem | 4×4 s4 (patchify) | Same |
| Downsampling | LN → 2×2 conv s2 | Same |

### Parameter Estimate (Base, no innovations)

```
Stage 1:                        120,768
  └─ Per block (C=96, e=3):     60,384 × 2
Stage 2:                        925,440
  └─ Per block (C=192, e=3):   231,360 × 4
Stage 3:                     12,000,000
  └─ Per block (C=384, e=4): 1,200,000 × 10
Stage 4:                      4,800,000
  └─ Per block (C=512, e=3): 1,600,000 × 3
                                    
Total blocks:                 17,846,208
Downsample 1 (96→192):            73,920
Downsample 2 (192→384):          295,296
Downsample 3 (384→512):          787,200
Stem:                              4,608
Head (LN+FC, 1000-class):        513,024
────────────────────────────────────────
Base total:                    19,520,256
```

---

## 2. Innovation 1: Multi-Receptive Field Depthwise Convolution (MRF-DW)

### Motivation

ConvNeXt V2 uses uniform 7×7 depthwise conv across all blocks. This ignores the principle that early stages (high spatial resolution) benefit from smaller receptive fields to capture fine details, while later stages (low spatial resolution) benefit from larger fields for global context. Multi-scale kernels within a single block improve feature diversity at negligible parameter cost.

### Design

Each block executes two parallel depthwise convolutions and mixes their outputs via learned per-channel weights:

```
Input (C-dim)
    │
    ├── DW-Conv (small kernel)
    ├── DW-Conv (base kernel)
    │
    └── α ⊙ small(x) + (1-α) ⊙ base(x)  →  Output
```

Where `α ∈ ℝ^C` is a learnable per-channel mixing vector initialized to 0.5.

**Kernel configuration by stage:**

| Stage | Base Kernel | Small Kernel | Effective Range | Extra Params |
|---|---|---|---|---|
| 1 (56×56) | 7×7 | 3×3 | 3–7 px | 9·C + C |
| 2 (28×28) | 7×7 | 3×3 | 3–7 px | 9·C + C |
| 3 (14×14) | 7×7 | 5×5 | 5–7 px | 25·C + C |
| 4 (7×7) | 7×7 | 5×5 | 5–7 px | 25·C + C |

### Parameter Overhead

| Stage | C | Blocks | Extra per Block | Total Extra |
|---|---|---|---|---|
| 1 | 96 | 2 | 9·96 + 96 = 960 | 1,920 |
| 2 | 192 | 4 | 9·192 + 192 = 1,920 | 7,680 |
| 3 | 384 | 10 | 25·384 + 384 = 9,984 | 99,840 |
| 4 | 512 | 3 | 25·512 + 512 = 13,312 | 39,936 |
| | | | **Total MRF-DW overhead** | **149,376** |

### Expected Behavior

- **Multi-scale feature diversity**: Parallel kernels capture complementary spatial scales
- **Learnable fusion**: The network can adaptively weight local vs. medium-range features per channel
- **Minimal overhead**: ~150K additional params (<1% of total model)

### Falsification Condition

If MRF-DW does not improve top-1 accuracy by ≥0.3% over the base configuration (same channel/depth budget with single 7×7 DW) on ImageNet-1K, the multi-scale mixing provides no measurable benefit over uniform receptive fields.

---

## 3. Innovation 2: Local-Enhanced GRN (LE-GRN)

### Motivation

GRN in ConvNeXt V2 normalizes using global statistics: `Gx = x · RMS(x)` where `RMS(x)` is computed over all spatial positions (H×W). This discards local contrast information. For models with fewer channels, each channel carries more semantically distinct information, and local contrast normalization can emphasize feature boundaries and texture.

### Design

LE-GRN augments GRN with a local mean subtraction step before the global normalization:

```
x_local = x - μ_local(x)            # Local mean removal (3×3 avg pool, padding=1)
Gx = x_local · RMS(x_local)         # Element-wise scaling by RMS
Nx = Gx / (RMS(Gx).mean(-1) + ε)    # Global normalization (over channel dim)
output = γ · Nx + β + x             # Learnable scale/shift + residual
```

Where:
- `μ_local(x)`: 3×3 average pooling (no padding → H-2×W-2, or padding=1 for same size)
- `γ, β ∈ ℝ^C`: Learnable parameters (same as original GRN)
- `ε = 1e-6`: Numerical stability

### Parameter Count

Same as original GRN: **2·C** per block (γ and β). The local mean subtraction adds zero trainable parameters (just a fixed avg-pool operation).

### Total LE-GRN Parameters Across Model

```
Stage 1: 2·96·2 = 384
Stage 2: 2·192·4 = 1,536
Stage 3: 2·384·10 = 7,680
Stage 4: 2·512·3 = 3,072
Total: 12,672
```

Identical to what GRN would cost — **zero net overhead**.

### Expected Behavior

- **Local contrast enhancement**: Subtracting local mean before global normalization emphasizes relative feature differences, similar to center-surround normalization in biological vision
- **Texture sensitivity**: Improved response to textured regions where local contrast is high
- **Same parameter cost**: No tradeoff in model size

### Falsification Condition

If LE-GRN does not improve top-1 accuracy by ≥0.2% over standard GRN (with identical base configuration) on ImageNet-1K, the local contrast enhancement provides no measurable benefit.

---

## 4. Innovation 3: Adaptive Expansion

### Motivation

ConvNeXt V2 uses uniform expansion ratio 4 across all stages. This is wasteful:
- **Stages 1–2** (high spatial resolution): The 1×1 convs are applied over 56×56 or 28×28 feature maps, dominating FLOPs. Reducing expansion here saves compute with minimal accuracy loss since early stages have fewer channels anyway.
- **Stage 3** (medium resolution, most blocks): Benefits from higher expansion because it does the bulk of semantic processing.
- **Stage 4** (low resolution, 7×7): Channel mixing at 7×7 spatial size is very cheap in FLOPs, but high C makes it expensive in params. Modest expansion is sufficient since stage 4 refines features rather than building new representations.

### Design

| Stage | Channels | Blocks | Expansion (V2-T) | Expansion (V2-E) | Rationale |
|---|---|---|---|---|---|
| 1 | 96 | 2 | ×4 | ×3 | High spatial res → save FLOPs |
| 2 | 192 | 4 | ×4 | ×3 | High spatial res → save FLOPs |
| 3 | 384 | 10 | ×4 | ×4 | Most blocks → maximize capacity |
| 4 | 512 | 3 | ×4 | ×3 | Low spatial res → refinement only |

### Parameter Savings vs. Uniform ×4

```
Stage 1: (4-3)·96·96·2 = 96²·2 = 18,432 saved
Stage 2: (4-3)·192·192·4 = 192²·4 = 147,456 saved
Stage 4: (4-3)·512·512·3 = 512²·3 = 786,432 saved
Total saved vs. uniform ×4: ≈952,320
```

This headroom is reinvested into the extra blocks in stage 3 (10 instead of 9) and the MRF-DW overhead.

---

## 5. Complete Model Parameter Budget

### Layer-by-Layer Breakdown
| **Stem** (3→96, 4×4 s4) | 4,608 | 0.02% |
| **Stage 1** (C=96, 2 blocks, e=3) | 120,768 | 0.6% |
| MRF-DW overhead (S1) | 1,920 | <0.01% |
| LE-GRN (S1) | 384 | <0.01% |
| **Stage 2** (C=192, 4 blocks, e=3) | 925,440 | 4.6% |
| MRF-DW overhead (S2) | 7,680 | 0.04% |
| LE-GRN (S2) | 1,536 | 0.01% |
| **Stage 3** (C=384, 10 blocks, e=4) | 12,000,000 | 59.8% |
| MRF-DW overhead (S3) | 99,840 | 0.5% |
| LE-GRN (S3) | 7,680 | 0.04% |
| **Stage 4** (C=512, 3 blocks, e=3) | 4,800,000 | 23.9% |
| MRF-DW overhead (S4) | 39,936 | 0.2% |
| LE-GRN (S4) | 3,072 | 0.02% |
| **Downsample 1** (LN+conv 96→192) | 73,920 | 0.4% |
| **Downsample 2** (LN+conv 192→384) | 295,296 | 1.5% |
| **Downsample 3** (LN+conv 384→512) | 787,200 | 4.0% |
| **Head** (LN + FC) | 513,024 | 2.6% |
| **────────────────────────────────** | | |
| | Component | Params | % of Total | Note |
|---|---|---|---|---|
| **Stem** (3→96, 4×4 s4) | 4,608 | 0.02% | |
| **Stage 1** (C=96, 2 blocks, e=3) | 120,768 | 0.6% | Includes GRN (192×2=384) |
| MRF-DW overhead (S1) | 1,920 | <0.01% | Extra over standard 7×7 DW |
| **Stage 2** (C=192, 4 blocks, e=3) | 925,440 | 4.7% | Includes GRN (384×4=1,536) |
| MRF-DW overhead (S2) | 7,680 | 0.04% | Extra over standard 7×7 DW |
| **Stage 3** (C=384, 10 blocks, e=4) | 12,000,000 | 60.9% | Includes GRN (768×10=7,680) |
| MRF-DW overhead (S3) | 99,840 | 0.5% | Extra over standard 7×7 DW |
| **Stage 4** (C=512, 3 blocks, e=3) | 4,800,000 | 24.4% | Includes GRN (1,024×3=3,072) |
| MRF-DW overhead (S4) | 39,936 | 0.2% | Extra over standard 7×7 DW |
| **Downsample 1** (LN+conv 96→192) | 73,920 | 0.4% | |
| **Downsample 2** (LN+conv 192→384) | 295,296 | 1.5% | |
| **Downsample 3** (LN+conv 384→512) | 787,200 | 4.0% | |
| **Head** (LN + FC) | 513,024 | 2.6% | |
| **────────────────────────────────** | | |
| **Total** | **19,669,632** | **100%** | |

---

## 6. FLOPs Estimate (224×224 input)

| Stage | Spatial Size | C | Blocks | FLOPs per Block | Total FLOPs |
|---|---|---|---|---|---|
| Stem | 224→56 | 3→96 | — | — | ~0.06G |
| Stage 1 | 56×56=3136 | 96 | 2 | DW(7×7): 2·3136·96·49 ≈ 29.5M<br>FC(×3): 2·3136·96·288 ≈ 173.4M | ~406M |
| Stage 2 | 28×28=784 | 192 | 4 | DW(7×7): 2·784·192·49 ≈ 14.8M<br>FC(×3): 2·784·192·576 ≈ 173.4M | ~753M |
| Stage 3 | 14×14=196 | 384 | 10 | DW(7×7): 2·196·384·49 ≈ 7.4M<br>FC(×4): 2·196·384·1536 ≈ 231.2M | ~2,386M |
| Stage 4 | 7×7=49 | 512 | 3 | DW(7×7): 2·49·512·49 ≈ 2.5M<br>FC(×3): 2·49·512·1536 ≈ 77.2M | ~239M |
| Head | 7×7=49 | 512 | — | — | ~1M |
| **Total** | | | | | **~3.8G FLOPs** |

**Comparison**: ConvNeXt V2-T ~4.5G → V2-E ~3.8G (**~16% reduction**)

MRF-DW adds ~4–8% extra FLOPs from the second DW branch (depthwise convs are memory-bound, so wall-clock impact is smaller than the FLOPs fraction suggests).

---

## 7. Training Protocol Recommendations

### Supervised Learning (ImageNet-1K)

| Hyperparameter | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (β₁=0.9, β₂=0.999) | Standard for ConvNeXt-style |
| Base LR | 4e-3 (cosine decay) | Lower LR for smaller model |
| Weight decay | 0.05 | Moderate regularization |
| Batch size | 4096 | Standard large-batch setting |
| Warmup epochs | 20 | Gradual ramp-up |
| Total epochs | 300 | Full training budget |
| Label smoothing | 0.1 | Improves generalization |
| DropPath rate | 0.1 (linear increase) | Regularization for depth |
| Data augmentation | RandAug(9, 0.5), Mixup(0.8), CutMix(1.0) | Standard modern pipeline |
| EMA | 0.9999 | Stabilizes training |

### FCMAE Self-Supervised Pretraining

| Hyperparameter | Value |
|---|---|
| Mask ratio | 0.6 (lower than ViT's 0.75) |
| Decoder | Sparse ConvNet (lightweight) |
| Pretraining epochs | 400 (medium-scale) |
| Optimizer | AdamW, LR=1.5e-4 |
| Weight decay | 0.05 |

---

## 8. Comparison with Existing Sub-20M Models

| Model | Params | FLOPs | Inductive Bias | Key Limitation |
|---|---|---|---|---|
| **EfficientNet-B1** | 7.8M | ~0.7G | Locality + SE | NAS architecture, limited modifiability |
| **MobileNetV3-L** | 5.4M | ~0.2G | Locality + SE | Mobile-optimized, saturates at higher res |
| **RegNetY-4GF** | 21M | 4.0G | Locality | Above 20M budget |
| **ConvNeXt V2-E** | **19.7M** | **~3.8G** | Multi-scale locality + local contrast | Proposed here |
| **ConvNeXt V2-T** | 28M | ~4.5G | Locality + GRN | Above 20M budget |
| **ResNet-50** | 25.6M | ~4.1G | Locality (3×3) | Outdated design, saturates |

ConvNeXt V2-E targets the **sweet spot** between efficient ConvNets and modernized designs, with the highest expected accuracy among sub-20M pure ConvNets due to its modern block design, multi-scale receptive fields, and enhanced normalization.
