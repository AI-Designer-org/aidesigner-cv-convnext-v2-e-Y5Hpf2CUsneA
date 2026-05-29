# ConvNeXt V2-E: Architecture Specification

## Domain Identification

| Domain | Relevance | Rationale |
|---|---|---|
| **Computer Vision (CV)** | Primary | Image classification backbone; dense prediction compatible |
| **GenAI** | Secondary | FCMAE self-supervised pretraining; diffusion decoder compatible |

---

## Upstream Research Contract Summary

Read from `/artifacts/j_Y5Hpf2CUsneA/work/research/03_novelty_gaps_and_lifecycle_contract.yaml`:

| Field | Value |
|---|---|
| Task level | `level_1` |
| Research question | Can a ConvNeXt V2 variant under 20M params match/exceed ConvNeXt V2-T (28M) representational capacity via three architectural innovations? |
| Novelty claims | (1) MRF-DW multi-scale depthwise mixing, (2) LE-GRN local-contrast normalization, (3) stage-adaptive expansion ratios |
| Baseline | ConvNeXt V2-T (28M), budget-matched V2 baseline, EfficientNet-B2/B3, Swin-T |
| Evaluation | Top-1 IN-1K, FCMAE transfer, ablation of each innovation, throughput/FLOPs/memory |
| Falsification | <80.0% top-1 supervised (300ep, 224×224) OR any innovation <0.15% improvement over ablation |

---

## ModelConfig

Defined in `01_model_config.py`. Key hyperparameters:

| Parameter | Value | Rationale |
|---|---|---|
| `dims` | [96, 192, 384, 512] | Stage 4 capped at 512 (not 768) to stay <20M |
| `depths` | [2, 4, 10, 3] | Bulk of capacity shifted to Stage 3 (10 blocks) |
| `expansion_ratios` | [3.0, 3.0, 4.0, 3.0] | Adaptive: high-res stages save params, mid-res gets max capacity |
| `use_mrf_dw` | True | Multi-scale depthwise mixing |
| `use_le_grn` | True | Local-enhanced normalization |
| `drop_path_rate` | 0.1 | Linear schedule across 19 blocks |
| `le_grn_local_kernel` | 3 | 3×3 local window for mean subtraction |

---

## Core Block: ConvNeXtV2EBlock

### Pseudocode

```python
def convnext_v2e_block(x, cfg):
    """
    x: (B, C, H, W)
    cfg: ModelConfig with per-stage fields (dim, expansion_ratio, small_kernel)
    Returns: (B, C, H, W)
    """
    shortcut = x

    # ── Innovation 1: MRF-DW ────────────────────────────────────────
    if cfg.use_mrf_dw:
        base_out = dw_conv7x7(x)                                 # (B, C, H, W)
        small_out = dw_conv_kxk(x, k=cfg.small_kernel)            # (B, C, H, W)
        alpha = sigmoid(mix_weight)                                # (1, C, 1, 1)
        x = alpha * base_out + (1.0 - alpha) * small_out          # learned per-channel mix
    else:
        x = dw_conv7x7(x)                                         # standard single kernel

    # ── Inverted bottleneck ─────────────────────────────────────────
    x = layer_norm_2d(x)                                           # (B, C, H, W)
    x = conv1x1(x, out_dim=cfg.hidden_dim)                         # expand: C → e·C
    x = gelu(x)                                                    # activation
    x = conv1x1(x, out_dim=cfg.dim)                                # project: e·C → C

    # ── Innovation 2: LE-GRN ───────────────────────────────────────
    if cfg.use_le_grn:
        # Local mean subtraction (3×3 avg pool)
        local_mean = avg_pool2d(x, k=3, pad=1, count_include_pad=False)
        x_local = x - local_mean                                   # local contrast
        Gx = norm(x_local, p=2, dim=(2,3))                         # global RMS of local-contrast features
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + cfg.grn_eps)     # normalize across channels
        x = gamma * (x * Nx) + beta + x                            # γ·(x·Nx) + β + residual
    else:
        # Standard GRN
        Gx = norm(x, p=2, dim=(2,3))
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + cfg.grn_eps)
        x = gamma * (x * Nx) + beta + x

    # ── Stochastic depth ───────────────────────────────────────────
    x = drop_path(x, cfg.drop_path_prob)

    return shortcut + x
```

### Key design decisions in the block

| Decision | Choice | Justification |
|---|---|---|
| Block ordering | DW → LN → PW1 → GELU → PW2 → GRN → DropPath | Matches ConvNeXt V2; spatial mixing in narrow channel space saves params+FLOPs |
| Pre-norm vs post-norm | Pre-norm (LN after DW) | LN before pointwise convs stabilizes activations at the wide channel expansion |
| MRF-DW placement | Before LN and PW convs | Spatial mixing in narrow space avoids expensive multi-scale at high dimension |
| LE-GRN location | After PW2 projection | GRN needs to operate on final channel-mixed features; matches original V2 placement |
| Sigmoid gate | α = σ(mix_weight) | Ensures convex combination α∈(0,1); gradient flows through both branches at init |
| Mix weight init | 0 (α=0.5) | Equal weighting at initialization; network learns asymmetric weighting if beneficial |

---

## Architecture Diagram (ASCII)

```
                     Input Image (B, 3, 224, 224)
                                │
                                ▼
                     ┌─────────────────────┐
                     │       Stem          │  Conv2D 4×4, stride 4
                     │  (3 → 96 ch)        │  + LayerNorm
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
          │  │  └───────────────────────────┘   │   │
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

---

## Parameter Distribution

| Component | Params | % of Total | Notes |
|---|---|---|---|
| Stem (3→96, 4×4 s4) | 4,608 | 0.02% | Negligible |
| **Stage 1** (C=96, 2 blks, e=3) | 122,688 | 0.6% | Includes MRF-DW overhead |
| **Stage 2** (C=192, 4 blks, e=3) | 933,120 | 4.7% | |
| **Stage 3** (C=384, 10 blks, e=4) | 12,099,840 | **61.5%** | Dominant — bulk of capacity |
| **Stage 4** (C=512, 3 blks, e=3) | 4,839,936 | **24.6%** | Second largest |
| Downsample 1 (96→192) | 73,920 | 0.4% | |
| Downsample 2 (192→384) | 295,296 | 1.5% | |
| Downsample 3 (384→512) | 787,200 | 4.0% | |
| Head (LN + FC 512→1000) | 513,024 | 2.6% | |
| **Total** | **19,669,632** | **100%** | |

**Under 20M budget by 330K params**. Headroom for minor additions or debugging hooks.

---

## Parameter Budget Breakdown (Per-Stage Formula)

```
Per-block formula:
  P_block(C, e) = 49·C  (DW 7×7)
                + 2·C   (LN γ,β)
                + e·C²  (PW1 expand)
                + e·C²  (PW2 project)
                + 2·C   (GRN γ,β)
                = 2·e·C² + 53·C

MRF-DW overhead:
  Δ_MRF(C, k_small) = C·(k_small² + 1)   [replaces 49C DW with (49C + k_small²·C + C)]

Stage total:
  P_stage = B·[2·e·C² + 53·C] + B·C·(k_small² + 1)   [if MRF-DW enabled]

Downsampling:
  P_down(C_in, C_out) = 2·C_in + C_in·C_out·4   [LN + 2×2 conv]

Head:
  P_head = 2·C_last + C_last·n_classes
```

---

## Inductive Bias Justification

Every non-standard design choice, one sentence each:

| Design Choice | Inductive Bias Statement |
|---|---|
| **MRF-DW (parallel 7×7 + smaller kernel)** | Multi-scale features within each block improve representational diversity at <1% parameter overhead compared to a single kernel, especially important when channel count is capped at 512. |
| **Learnable per-channel mixing weight** | Different channels should weigh local vs. medium-range spatial context differently — a uniform mixing ratio across all channels is suboptimal. |
| **Sigmoid-gated mixing (α = σ(w))** | Ensures a convex combination α∈(0,1), preventing either kernel from being entirely suppressed while allowing the network to smoothly vary the mix per channel. |
| **LE-GRN (local mean subtraction before global RMS)** | Local contrast (edges, textures) provides complementary signal to global channel statistics; subtracting local mean before normalization boosts channels with high spatial variance without adding parameters. |
| **Adaptive expansion [3,3,4,3] instead of uniform ×4** | High-resolution stages (1,2) save FLOPs/params with lower expansion since they process many spatial locations; the mid-resolution stage (3) benefits from full ×4 capacity; the low-resolution final stage (4) needs only moderate expansion for refinement. |
| **Pre-norm (LN after DW before PW)** | Normalizing activations before the expensive pointwise expansion prevents activation magnitudes from growing unbounded through the bottleneck, stabilizing training at any depth. |
| **Depthwise conv before pointwise expansion** | Spatial mixing in the narrow (C-dim) space is parameter-efficient; expanding channels before spatial mixing would incur 4× the parameter cost for the DW conv. |
| **Stem: single 4×4 stride-4 conv** | Aggressive early downsampling creates a 56×56 feature map where subsequent blocks operate efficiently, matching the ResNet/ConvNeXt convention. |
| **Downsampling: LN → 2×2 stride-2 conv** | Separating normalization from spatial reduction avoids the aliasing and optimization issues of strided convolutions on un-normalized features. |

---

## Research-to-Architecture Traceability

Each non-standard architectural decision maps to a research contract item:

| Research Contract Item | Architecture Decision | Evidence Status | Validation Hook |
|---|---|---|---|
| **Gap 1**: Fixed receptive field in ConvNeXt V2 blocks | MRF-DW: parallel 7×7 + small-kernel depthwise conv with learned mixing | `hypothesis` | Ablate: `use_mrf_dw=False`, measure Δtop-1. If <0.3%, claim falsified. |
| **Gap 2**: GRN lacks local contrast | LE-GRN: local mean subtraction (3×3 avg pool) before global RMS normalization | `hypothesis` | Ablate: `use_le_grn=False` (fallback to standard GRN), measure Δtop-1. If <0.2%, claim falsified. |
| **Gap 3**: Suboptimal stage capacity at small scale | Adaptive expansion [3,3,4,3] vs uniform ×4; depth redistribution [2,4,10,3] vs [3,3,9,3] | `hypothesis` | Compare `expansion_ratios=[4,4,4,4]` vs `[3,3,4,3]`. Measure Δparams + Δtop-1. |
| **Baseline**: ConvNeXt V2-T (28M, 82.8%) | V2-E targets 19.7M with innovations to close accuracy gap | `grounded`: V2-T is published result | Train V2-E supervised 300ep; compare top-1 vs 82.0% expected V2-T (supervised) |
| **Evaluation**: Throughput on A100 / FLOPs | MRF-DW adds 4–8% FLOPs (depthwise convs are memory-bound, less wall-clock impact) | `TODO: unverified` | Profile images/sec on A100 at batch=256, compare to budget-matched baseline |
| **FCMAE compatibility**: Does LE-GRN affect SSL? | LE-GRN replaces GRN at identical param count and position; same interface | `TODO: unverified` | Pretrain V2-E with FCMAE 400ep, fine-tune, compare to supervised-only result |
| **Blocking unknown**: Uniform downscaling baseline | Config `dims` [96,192,384,512] vs hypothetical uniform-downscaled [80,160,320,640] | `hypothesis` | Construct and train uniform-downscaled V2-T at ~19.7M; compare to V2-E with innovations |

---

## CV Domain-Specific Considerations

| Concern | How ConvNeXt V2-E Addresses It |
|---|---|
| **Spatial handling at different resolutions** | Hierarchical 4-stage design (1/4 → 1/8 → 1/16 → 1/32) compatible with arbitrary input sizes via padding. Global avg pool at head removes FC resolution dependence. |
| **Multi-scale features** | Explicit multi-scale via pyramid stages. MRF-DW adds within-block multi-scale. Compatible with FPN/neck for detection/segmentation. |
| **Dense vs global processing** | Fully convolutional — operates on all spatial positions in parallel. No quadratic attention. 7×7 DW receptive field covers 49 spatial positions locally but grows to (1+6·B)×(1+6·B) effective through stack of B blocks. |
| **Translation equivariance** | Depthwise convolutions are translation-equivariant; global avg pool removes spatial dependence at head |
| **Scale invariance** | Not built-in (no explicit multi-scale test-time augmentation). Standard practice: test at multiple resolutions and average predictions. |
| **Inductive bias vs. data hunger** | Strong locality inductive bias (convolution) means smaller models converge faster with less data than ViTs. Appropriate for ImageNet-1K scale. |
| **Multi-scale at block level (MRF-DW)** | Early stages (56×56, 28×28): small kernel=3×3 captures fine detail (edges, corners). Late stages (14×14, 7×7): small kernel=5×5 provides wider context since spatial resolution is already low. |
| **Normalization choice** | LayerNorm (not BatchNorm) avoids batch-size dependence and matches ConvNeXt V2. Batch-size-independent: works with any batch size including 1. |
| **Stochastic depth** | Linear DropPath schedule (0 → max) across 19 blocks provides stronger regularization for deeper blocks while allowing early blocks to train fully. |

---

## Implementation Risk Flags

| Risk | Severity | Mitigation | Contingency |
|---|---|---|---|
| **MRF-DW sigmoid gate saturation**: If the sigmoid weights saturate to 0 or 1 for all channels, the second kernel provides no benefit, wasting 150K params | Low | Initialize `mix_weight=0` (α=0.5). Monitor α distribution during training — if std(α) < 0.05 after 50 epochs, MRF-DW is not being used | Replace sigmoid with softmax (α₁, α₂) which always keeps both branches active |
| **LE-GRN edge artifacts**: `avg_pool2d` with `count_include_pad=False` at borders has fewer effective pixels for local mean, causing boundary effects | Medium | Use reflection padding mode instead of zero padding in the avg pool. Or pad with replicated borders | Fall back to standard GRN; the local contrast step adds zero parameters so there is no sunk cost |
| **Stage 3 parameter dominance**: 61.5% of params in Stage 3 creates a bottleneck in gradient flow and optimizer updates | Medium | Monitor Stage 3 gradient norms vs. other stages. If variance is high, adjust DropPath schedule to be stage-aware | Shift one block from Stage 3 to Stage 4 (B=[2,4,9,4]) to redistribute capacity |
| **DW conv memory-bound overhead**: The extra small-kernel DW conv doubles the memory-bandwidth-bound depthwise ops, potentially hurting throughput more than the FLOPs fraction suggests | High — most practical concern | Benchmark throughput early. DW convs benefit from fused implementations (TensorRT, PyTorch 2.0 compile) | Drop MRF-DW and use standard 7×7; or use a single dilated 7×7 conv with learned dilation per channel |
| **FCMAE compatibility**: LE-GRN's local mean subtraction might interact poorly with masked features (zeroed-out regions in masked autoencoding) | Medium | Test FCMAE pretraining early (100ep probe). If LE-GRN degrades SSL, use standard GRN during pretrain and LE-GRN during fine-tune | Use GRN for FCMAE stage; fine-tune with LE-GRN since the head is re-initialized anyway |

---

## Suggested Ablations

All ablations are expressible as single-field `ModelConfig` changes.
Ordered by "turn this off first if it doesn't work":

| # | Ablation | Config Field | Baseline Value | Ablated Value | Hypothesis Tested | Expected Metric Movement | Failure Interpretation | Owning Stage |
|---|---|---|---|---|---|---|---|---|
| 1 | **Remove MRF-DW** | `use_mrf_dw` | `True` | `False` | MRF-DW multi-scale mixing improves top-1 ≥0.3% over single 7×7 at same param budget | Δtop-1 ≥ -0.3% (i.e., 0.3% drop when removed) | If Δtop-1 < 0.15%: MRF-DW does not justify its 4–8% FLOPs overhead. Direct to `ml-research` for Gap 1 re-evaluation. | `ml-research` |
| 2 | **Remove LE-GRN** | `use_le_grn` | `True` | `False` | LE-GRN local contrast improves top-1 ≥0.2% over standard GRN at zero param cost | Δtop-1 ≥ -0.2% | If Δtop-1 < 0.15%: local mean subtraction provides negligible benefit. Revert to standard GRN permanently. | `ml-architect` |
| 3 | **Uniform expansion ×4** | `expansion_ratios` | `[3,3,4,3]` | `[4,4,4,4]` | Adaptive expansion saves ~0.95M params with ≤0.1% accuracy loss vs uniform ×4 | Δparams ≤ -0.95M, Δtop-1 ≥ -0.1% | If uniform ×4 is >0.3% better at same param budget: adaptive expansion is harming capacity. | `ml-architect` |
| 4 | **Swap depth distribution** | `depths` | `[2,4,10,3]` | `[3,3,9,3]` | Shifting one block from S3→S1 improves gradient flow without accuracy loss | ≤0.1% change in top-1 | If >0.2% drop: the [2,4,10,3] distribution is optimal for this budget. | `ml-architect` |
| 5 | **LE-GRN kernel size 5×5** | `le_grn_local_kernel` | `3` | `5` | Larger local window captures more meaningful local context for contrast enhancement | Δtop-1 ≥ +0.05% vs 3×3 | If 5×5 ≤ 3×3: smaller window is sufficient; keep 3×3 for lower FLOPs. | `ml-architect` |
| 6 | **MRF-DW small kernel 5×5 in S1-2** | `small_kernel_sizes` | `[3,3,5,5]` | `[5,5,5,5]` | Wider small kernel in early stages captures more useful context at high resolution | Δtop-1 ≥ +0.1% vs [3,3,5,5] | If ≤0.05%: early stages benefit from fine-grained 3×3 details; keep [3,3,5,5]. | `ml-architect` |
| 7 | **FCMAE pretraining (full pipeline)** | (training config) | Supervised 300ep | FCMAE 400ep + fine-tune 100ep | FCMAE pretraining provides ≥1.0% gain over supervised-only | Δtop-1 ≥ +1.0% vs supervised-only | If <0.5% gain: LE-GRN or MRF-DW may interfere with masked pretraining. Investigate. | `ml-research` |
| 8 | **Uniform channel downscale (strongest baseline)** | `dims` | `[96,192,384,512]` | `[80,160,320,640]` | [80,160,320,640] at uniform ×4 matches V2-E's param budget without innovations | Δtop-1 within 0.3% of V2-E | If [80,160,320,640] ≥ V2-E: innovations do not justify complexity. This is the blocking unknown. | `ml-research` |

---

## Evaluation Requirements Carried Forward

From lifecycle contract, the validator/trainer must execute:

1. **Primary**: Top-1 accuracy on ImageNet-1K validation (224×224), supervised 300 epochs
2. **Secondary**: Top-1 with FCMAE pretraining (400ep + 100ep fine-tune)
3. **Ablations**: Each single-field change from the table above, identical training schedule
4. **Efficiency**: Throughput (images/sec) on single A100 at batch=256
5. **Memory**: Peak GPU memory at batch=256 (train + inference)
6. **FLOPs**: Count per forward pass at 224×224

### Baseline Requirements

| Baseline | Params | Source | Comparison |
|---|---|---|---|
| ConvNeXt V2-T (supervised) | 28M | Published (82.0% top-1 expected) | Accuracy-per-parameter ratio |
| ConvNeXt V2 budget-matched (ablated baseline) | ~19.5M | `ablated_baseline_config()` | Isolate innovation contributions |
| EfficientNet-B2 | 9.2M | Published (80.1%) | Efficient ConvNet competitor |
| EfficientNet-B3 | 12M | Published (81.7%) | Efficient ConvNet competitor |
| Swin-T | 28M | Published (81.3%) | Hierarchical ViT at higher budget |
| RegNetY-4GF | 21M | Published (~80.0%) | Design-space competitor |

---

## FLOPs Budget

| Stage | Spatial | C | Blocks | FLOPs |
|---|---|---|---|---|
| Stem | 224→56 | 3→96 | — | ~0.06G |
| Stage 1 | 56×56 | 96 | 2 | ~0.41G |
| Stage 2 | 28×28 | 192 | 4 | ~0.75G |
| Stage 3 | 14×14 | 384 | 10 | ~2.39G |
| Stage 4 | 7×7 | 512 | 3 | ~0.24G |
| Head | 7×7 | 512 | — | ~0.001G |
| **Total** | | | | **~3.8G** |

**MRF-DW overhead**: ~4–8% additional FLOPs from the second depthwise branch. Depthwise convs are memory-bandwidth-bound; the wall-clock impact is typically smaller than the FLOPs fraction suggests.
