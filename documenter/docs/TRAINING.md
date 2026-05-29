# Training & Reproduction

> **Note**: No training runs have been executed. This document describes the recommended training protocol designed by the research and architect stages. All accuracy claims in this document are extrapolated and marked `TODO: unverified`.

---

## Environment

Recommended environment for ImageNet-1K training:

- Python: 3.10+
- PyTorch: 2.0+ (tested with 2.5)
- CUDA: 11.8+, tested on NVIDIA A100 (80GB)
- Other: `fvcore` (for FLOPs counting), `timm` (for external baselines)

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install fvcore timm
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## Default hyperparameters

### Model configuration (`ModelConfig`)

| Field | Default | Rationale |
|---|---|---|
| `dims` | [96, 192, 384, 512] | Stage 4 capped at 512 (not V2-T's 768) to stay under 20M |
| `depths` | [2, 4, 10, 3] | Bulk of capacity in Stage 3 (10 blocks, 14×14 spatial) |
| `expansion_ratios` | [3.0, 3.0, 4.0, 3.0] | Adaptive: ×4 only where it matters most (Stage 3) |
| `use_mrf_dw` | True | Multi-scale depthwise mixing |
| `small_kernel_sizes` | [3, 3, 5, 5] | 3×3 for high-resolution early stages, 5×5 for low-resolution late stages |
| `use_le_grn` | True | Local-enhanced normalization at zero param cost |
| `le_grn_local_kernel` | 3 | 3×3 local window for mean subtraction |
| `drop_path_rate` | 0.1 | Linear stochastic depth schedule |
| `mrf_mix_init` | 0.0 | Sigmoid(0) = 0.5 — equal branch weighting at initialization |
| `base_kernel_size` | 7 | Matches ConvNeXt V2 base kernel |

### Training hyperparameters (`TrainingConfig`)

| Field | Default | Rationale |
|---|---|---|
| `optimizer` | AdamW | Standard for ConvNeXt-style architectures |
| `base_lr` | 4e-3 | Lower than V2-T (4e-3 vs typical 1e-3 for ConvNeXt) — smaller model needs lower LR |
| `weight_decay` | 0.05 | Moderate regularization for sub-20M model |
| `beta1, beta2` | (0.9, 0.999) | Standard AdamW defaults for ConvNeXt |
| `scheduler` | cosine decay | Standard cosine schedule with linear warmup |
| `warmup_epochs` | 20 | Gradual ramp-up for stable early training |
| `total_epochs` | 300 | Full supervised training budget |
| `batch_size` | 4096 | Standard large-batch setting (distributed across 8 GPUs) |
| `label_smoothing` | 0.1 | Improves generalization |
| `mixup_alpha` | 0.8 | Mixup augmentation strength |
| `cutmix_alpha` | 1.0 | CutMix augmentation strength |
| `randaug_magnitude` | 9 | RandAugment magnitude (9 out of 15) |
| `randaug_num_ops` | 2 | Number of RandAugment operations per image |
| `ema_decay` | 0.9999 | Exponential moving average of weights |

---

## Recommended training recipe

| Setting | Value | Notes |
|---|---|---|
| Optimizer | AdamW | β₁=0.9, β₂=0.999 |
| Peak LR | 4e-3 | Linear warmup over 20 epochs, cosine decay to 1e-6 |
| Batch size | 4096 | 8×A100 (512 per GPU) or gradient accumulation |
| Weight decay | 0.05 | Excluded from bias and LayerNorm parameters |
| Grad clip | None (global norm if needed) | ConvNeXt training typically does not use grad clipping |
| Precision | bf16 mixed | fp32 master weights; LayerNorm casts reduction to fp32 |
| Augmentation | RandAug(9, 0.5), Mixup(0.8), CutMix(1.0) | Standard modern augmentation pipeline |
| Label smoothing | 0.1 | Applied in cross-entropy loss |
| DropPath | 0 → 0.1 (linear) | Per-block schedule; linear increase across 19 blocks |
| EMA | 0.9999 | Applied at end of each epoch |

### Domain-specific CV training notes

- **Token batch size**: 4096 images per step distributed across GPUs
- **Input resolution**: 224×224 with RandomResizedCrop + RandomHorizontalFlip
- **EMA teacher**: Maintain separate EMA copy of weights; use for validation
- **Learning rate scaling**: Linear scaling rule: lr = base_lr × batch_size / 1024 (gives 4e-3 × 4096/1024 = 1.6e-2 → adjusted to 4e-3 per architect)
- **Normalization**: Standard ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### FCMAE self-supervised pretraining (not yet implemented)

| Hyperparameter | Value | Rationale |
|---|---|---|
| Mask ratio | 0.6 | Lower than ViT's 0.75 — ConvNets need more visible context |
| Decoder | Sparse ConvNet (lightweight) | Lightweight reconstruction head |
| Pretraining epochs | 400 | Medium-scale schedule |
| Optimizer | AdamW, LR=1.5e-4 | Lower LR for SSL training |
| Weight decay | 0.05 | Match supervised setting |

---

## Expected behavior

> **TODO: unverified** — No training runs have been performed. The following are extrapolated estimates from the architect's analysis:

| Setting | Expected Top-1 (IN-1K, 224×224, 300ep supervised) |
|---|---|
| ConvNeXt V2-T (28M) — supervised | ~82.0% (published reference) |
| V2-E full (19.7M, all innovations) | **80.5–81.5%** |
| V2-E ablated base (no MRF-DW, no LE-GRN) | 80.2–81.0% |
| V2-E + FCMAE pretraining | **81.5–82.5%** |
| V2-E uniform ×4 (20.5M, over budget) | 80.5–81.5% (same accuracy, more params) |

**Key efficiency comparison** (if estimates hold):
- V2-T: 82.8% (FCMAE) / 28M = 2.96% / M
- V2-E: 81.5% / 19.7M = 4.14% / M → **40% better accuracy-per-parameter ratio**

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss NaN in first steps | bf16 float-sensitive op in LayerNorm or GRN | Verify `LayerNormNd` casts to fp32 for reduction; GRN/L2-norm already uses fp32 cast |
| MRF-DW mixing weight α saturates at 0 or 1 for all channels | Sigmoid gate saturation from large `mix_weight` values | Monitor α distribution. If std(α) < 0.05 after 50 epochs, replace sigmoid with softmax (ensures both branches always active) |
| LE-GRN degrades accuracy vs standard GRN | Local mean subtraction introduces noise for low-contrast features | Set `use_le_grn=False` and fall back to standard GRN; no sunk cost |
| Stage 3 gradient norms much smaller than other stages | 10-block depth in Stage 3 causes gradient attenuation | Adjust DropPath schedule to be stage-aware; consider shifting one block from Stage 3 to Stage 4 (B=[2,4,9,4]) |
| Throughput lower than expected vs ablated baseline | Second depthwise DW conv in MRF-DW doubles memory-bandwidth-bound ops | Benchmark with `torch.compile`; consider dropping MRF-DW for deployment |
| FCMAE self-supervised pretraining diverges | LE-GRN interacts poorly with zeroed-out masked regions | Use standard GRN during pretraining, LE-GRN only during fine-tune |
| Memory OOM at batch=4096 | Large activations from high-resolution stages | Enable gradient checkpointing (`use_checkpoint=True`) or reduce per-GPU batch size with gradient accumulation |
| Model accuracy worse than naive downscaled V2-T | Innovations do not justify complexity at this budget | This is the blocking unknown — train the uniform downscale baseline [80,160,320,640] and compare |
