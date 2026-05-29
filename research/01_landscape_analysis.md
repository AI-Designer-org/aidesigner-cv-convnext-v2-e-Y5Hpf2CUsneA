# ConvNeXt V2-E: Landscape Analysis

## Domain Identification

| Domain | Relevance | Rationale |
|---|---|---|
| **Computer Vision (CV)** | Primary | Image classification, dense prediction, backbone design |
| **Generative AI** | Secondary | FCMAE self-supervised pretraining; compatible with diffusion decoders |

---

## 1. ConvNeXt V2 Architecture Analysis

**Paper**: Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (CVPR 2023)

### Block Design (V2)

```
Input (C-dim)
    │
    ├── Depthwise Conv2D 7×7 → 49·C params
    ├── LayerNorm → 2·C params
    ├── Conv2D 1×1 (expand, ×4) → 4·C² params
    ├── GELU activation
    ├── Conv2D 1×1 (project) → 4·C² params
    ├── Global Response Normalization (GRN) → 2·C params
    ├── DropPath + residual
    │
    Output (C-dim)
```

**Per-block parameter formula** (expansion ratio `e`):
```
P_block = 49·C + 2·C + e·C² + e·C² + 2·C
        = (8·e)·C² + 53·C
```

For `e=4`: `P_block = 32·C² + 53·C`

### Model Variant Comparison

| Variant | C | Blocks | Expansion | Params | Top-1 IN-1K |
|---|---|---|---|---|---|
| ConvNeXt V2-T | [96,192,384,768] | [3,3,9,3] | 4 | 28M | 82.8%* |
| ConvNeXt V2-S | [96,192,384,768] | [3,3,27,3] | 4 | 50M | 83.9%* |
| ConvNeXt V2-B | [128,256,512,1024] | [3,3,27,3] | 4 | 89M | 84.9%* |
| ConvNeXt V2-L | [192,384,768,1536] | [3,3,27,3] | 4 | 198M | 85.1%* |

*\*With FCMAE pretraining*

### Key Design Decisions in V2 (vs V1)

| Component | ConvNeXt V1 | ConvNeXt V2 | Impact |
|---|---|---|---|
| Stem | 4×4 conv stride 4 | Same | Aggressive early downsampling |
| Normalization | LayerNorm (post-DW) | Same | Stablizes training |
| Activation | GELU | Same | Non-saturating nonlinearity |
| Feature competition | None | GRN after project | Prevents feature collapse in FCMAE |
| Pretraining | Supervised (IN-1K) | FCMAE (self-supervised) | Improves representation quality |
| Downsampling | LN → 2×2 conv stride 2 | Same | Separate spatial reduction |

### Parameter Distribution (ConvNeXt V2-T, 28M)

```
├── Stem (4×4 conv, stride 4):         4.6K   (0.02%)
├── Stage 1 (C=96, 3 blocks):         236K   (0.8%)
├── Stage 2 (C=192, 3 blocks):        915K   (3.3%)
├── Stage 3 (C=384, 9 blocks):       10.8M   (38.6%)
├── Stage 4 (C=768, 3 blocks):       14.3M   (51.0%)
├── Downsampling layers (×3):          1.6M   (5.5%)
├── Head (LN + classifier):            0.2M   (0.7%)
```

> **Key insight**: Stage 4 dominates (51% of params) for the T variant due to its 768 channels and expansion-4 MLP. Stage 3 is next at 39%. Any efficiency improvement must target these two stages.

---

## 2. Related Architecture Families

### Modernized ConvNets

| Architecture | Core Ops | Param Efficiency | Inductive Bias | Limitation |
|---|---|---|---|---|
| **ConvNeXt V2** | DW-7×7 + inverted bottleneck + GRN | Good (8·C²/block) | Translation equiv., locality | Fixed receptive field per block |
| **FocalNet** | Focal modulation (context aggregation) | Moderate (window-based) | Hierarchical context | Complex modulation gate |
| **EfficientNet** | MBConv + SE + compound scaling | Excellent (NAS-optimized) | Locality + channel attention | NAS-determined, less flexible |
| **MobileNetV3** | Depthwise sep. + SE + hard-swish | Excellent (mobile-targeted) | Locality | Performance ceiling at scale |
| **ConvNeXt V1** | DW-7×7 + inverted bottleneck | Good | Locality | Feature collapse in SSL |

### Vision Transformers

| Architecture | Core Ops | Param Efficiency | Inductive Bias | Limitation |
|---|---|---|---|---|
| **ViT** | Global self-attention, MLP | Good at scale | Minimal (patch-level) | Quadratic in patches, needs data |
| **Swin** | Window attention + shifted | Good | Locality + hierarchy | Window boundary artifacts |
| **DeiT** | ViT + distillation | Good | Teacher-dependent | Still quadratic attention |
| **EVA-02** | ViT + scaled ReLU, SwiGLU | Excellent | Minimal | Large model only benefits |

### Efficient Architectures (<20M)

| Architecture | Params | Top-1 IN-1K | Design Principle |
|---|---|---|---|
| **EfficientNet-B0** | 5.3M | 77.1% | NAS compound scaling |
| **MobileNetV3-L** | 5.4M | 75.2% | NAS + net adaptation |
| **ConvNeXt V2-Nano** | ~15M* | ~78%* | Downsized from T |
| **ViT-S** | 22M | 79.8% (AugReg) | Small ViT |
| **Swin-T** | 28M | 81.3% | Hierarchical attention |

*\*Estimated; ConvNeXt V2 does not officially define a Nano variant.*

---

## 3. Complexity & Properties

### Core Operator Comparison

| Operator | Time Complexity | Space Complexity | GPU Utilization | Memory-Bound? |
|---|---|---|---|---|
| **DW 7×7** | O(H·W·C·k²) | O(C·k²) | Poor (low compute intensity) | Yes |
| **DW 3×3** | O(H·W·C·9) | O(9·C) | Poor | Yes |
| **Conv 1×1 (pointwise)** | O(H·W·C₁·C₂) | O(C₁·C₂) | Excellent (matmul-like) | No (compute-bound) |
| **GRN** | O(H·W·C) | O(C) | Excellent (element-wise) | No |
| **LayerNorm** | O(H·W·C) | O(2·C) | Moderate | Yes (needs reduction) |

### Properties Summary

| Property | ConvNeXt V2-T | Proposed V2-E | Improvement |
|---|---|---|---|
| FLOPs (224×224) | ~4.5G | ~2.5G (est.) | ~44% reduction |
| Params | 28M | <20M | >28% reduction |
| Parallelism | Fully conv | Fully conv | Same (strong) |
| Memory footprint | Moderate | Low | Better for edge |
| Hardware alignment | Good (fused DW conv in PyTorch/TensorRT) | Same + better cache utilization | Improved |
| Length generalization | N/A (image) | N/A | N/A |
| Multi-scale features | 4 stages (1/4 → 1/32) | 4 stages (1/4 → 1/32) | Same |

---

## 4. ConvNeXt V2 Design Space Under 20M

### Scaling Strategy Comparison

Given the constraint of <20M parameters, three levers are available:

1. **Channel width (C)**: Dominant quadratic effect via 1×1 convs (O(C²) per block)
2. **Depth (B)**: Linear scaling in block count
3. **Expansion ratio (e)**: Linearly scales the 1×1 conv parameters

**Tradeoff**: For a fixed param budget, reducing C by 10% saves ~19% in the quadratic term but reduces per-channel capacity. Increasing depth compensates but adds FLOPs at high spatial resolutions.

### Identified Gaps

1. **Uniform receptive field** — Every block uses 7×7 DW regardless of stage. Early stages benefit from smaller fields (less parameter waste), later stages from larger fields (context aggregation).

2. **GRN lacks local contrast** — Global normalization ignores local feature competition, which is more important at smaller channel counts where each channel encodes distinct features.

3. **Fixed expansion ratio (×4)** — Standard from Swin-T, but suboptimal at small scale where channel-mixing capacity is already limited by narrow C.

4. **No cross-block feature reuse** — Each block operates independently without lightweight connections that could improve gradient flow in deeper small models.

5. **Downsampling is parameter-heavy** — The 2×2 stride-2 conv transitions between stages are expensive at large-to-large channel jumps (e.g., 384→768 costs ~1.2M params).

These gaps inform the proposed ConvNeXt V2-E design in the companion architecture document.
