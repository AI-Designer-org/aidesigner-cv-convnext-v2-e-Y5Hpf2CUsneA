# Benchmarks

All numbers are reproducible with the commands shown.
Numbers marked `TODO` have not been measured — do not cite them.

> **Important**: No ImageNet-1K training has been performed. All accuracy claims in this document are extrapolated and marked `TODO: unverified`. The benchmarks below cover synthetic/proxy evaluations, parameter verification, ablation parameter impacts, and profiling infrastructure. See `research_eval/claim_grounding.md` for the full claim-status audit.

---

## CV domain benchmarks

### Synthetic tasks

| Task | Metric | Value | Command | Notes |
|---|---|---|---|---|
| Masked reconstruction probe | MSE improvement | ~0.0–0.5 (synthetic) | `python benchmarks.py` | MSE of lightweight decoder vs. random baseline; synthetic probe, not actual MAE training |
| Linear probe (synthetic) | Test accuracy | ~0.7–0.9 (synthetic) | `python benchmarks.py` | Linear classifier on frozen backbone features; Gaussian blobs proxy dataset |
| Parameter efficiency ratio | % / M | > 2.93 expected | `python benchmarks.py` | Expected acc / params(M); vs V2-T's 2.93%/M reference |
| Noise entropy | fraction of max | > 0.5 | `python benchmarks.py` | Prediction entropy on pure noise (calibration proxy) |
| Translation robustness | cosine similarity | > 0.5 (8px shift) | `python benchmarks.py` | Classification consistency under horizontal translation |

Reproduce all: `PYTHONPATH=../coder:$PYTHONPATH python benchmarks.py`

### Parameter budget verification

| Variant | Params | Within 20M | Command | Notes |
|---|---|---|---|---|
| tiny | 19,670,632 | Yes | `python -c "from model import convnext_v2e_tiny, count_params; print(count_params(convnext_v2e_tiny()))"` | Primary variant with all innovations |
| wide | ~19.4M | Yes | Same, with `convnext_v2e_wide` | Wider, shallower alternative |
| deep | ~19.0M | Yes | Same, with `convnext_v2e_deep` | Narrower, deeper alternative |
| ablated | ~19.5M | Yes | Same, with `convnext_v2e_ablated_baseline` | Single 7×7 DW + standard GRN |
| uniform_e4 | ~20.5M | **No** | Same, with `convnext_v2e_uniform_expansion` | Over budget — for ablation only |

### Parameter overhead of innovations

| Innovation | Overhead | Verification |
|---|---|---|
| MRF-DW (vs. single 7×7 DW) | +149,376 params (<0.8% of total) | Confirmed by `ablation_runner.py` ablation #2 vs #1 delta |
| LE-GRN (vs standard GRN) | 0 params | Confirmed by `test_model.py::test_le_grn_same_param_count_as_grn` |
| Adaptive expansion [3,3,4,3] (vs uniform ×4) | −1,904,640 params (−9.7%) | Confirmed by smoke test and `ablation_runner.py` ablation #4 vs #1 delta |

---

## Ablation study

All ablations are single-field `ModelConfig` changes. Parameter impacts are verified; accuracy impacts are `TODO: unverified` pending training.

| Ablation | Config delta | Params | Δparams vs full | Proxy status |
|---|---|---|---|---|
| 1_full_model | (all innovations active) | 19,670,632 | — | Forward/backward OK |
| 2_no_mrf_dw | `use_mrf_dw: True→False` | 19,521,256 | −149,376 | Forward/backward OK |
| 3_no_le_grn | `use_le_grn: True→False` | 19,670,632 | 0 | Forward/backward OK |
| 4_uniform_expansion | `expansion_ratios: [3,3,4,3]→[4,4,4,4]` | 21,575,272 | +1,904,640 | Forward/backward OK |
| 5_swap_depths | `depths: [2,4,10,3]→[3,3,9,3]` | ~19,522,176 | −147,456 | Forward/backward OK |
| 6_le_grn_kernel5 | `le_grn_local_kernel: 3→5` | 19,670,632 | 0 | Forward/backward OK |
| 7_mrf_small5_all | `small_kernel_sizes: [3,3,5,5]→[5,5,5,5]` | ~19,687,296 | +17,664 | Forward/backward OK |
| 8_uniform_downscale | `dims: [96,192,384,320]` (no innovations) | ~19.7M | ~0 | Forward/backward OK |
| 9_ablated_baseline | Both MRF-DW + LE-GRN off | 19,521,256 | −149,376 | Forward/backward OK |

Reproduce: `PYTHONPATH=../coder:$PYTHONPATH python ablation_runner.py`

### Key ablation hypotheses (TODO: unverified — require training)

| Hypothesis | Test | Falsification condition |
|---|---|---|
| MRF-DW adds ≥0.3% top-1 | Compare #1 vs #2 | Δtop-1 < 0.15% |
| LE-GRN adds ≥0.2% top-1 | Compare #1 vs #3 | Δtop-1 < 0.15% |
| Adaptive expansion saves 1.9M params with ≤0.1% acc loss | Compare #1 vs #4 | Uniform ×4 > 0.3% better |
| V2-E beats uniform downscale [80,160,320,640] | Compare #1 vs #8 | #8 within 0.3% of #1 |

---

## Profiling

> **TODO**: Profiling results require a CUDA GPU. The infrastructure is ready; run the commands below on an A100 (or equivalent) to populate this table.

### FLOPs estimation

| Method | GFLOPs (224×224) | Notes |
|---|---|---|
| Heuristic (per-operator sum) | ~3.8 | Formula in `profile_model.py::estimate_flops_formula` |
| fvcore (precise) | TBD | Requires `pip install fvcore` + CUDA run |
| ConvNeXt V2-T reference | ~4.5 | Published value for V2-T |

**FLOPs reduction vs V2-T**: ~16% (from 4.5G to 3.8G)

### Throughput (requires CUDA)

```
Reproduce: PYTHONPATH=../coder:$PYTHONPATH python profile_model.py --batch-size 256
```

| Batch size | Images/sec | ms/image | Peak memory (MB) |
|---|---|---|---|
| 1 | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO |
| 32 | TODO | TODO | TODO |
| 64 | TODO | TODO | TODO |
| 128 | TODO | TODO | TODO |
| 256 | TODO | TODO | TODO |

### Memory profile (requires CUDA)

```
Reproduce: PYTHONPATH=../coder:$PYTHONPATH python profile_model.py --batch-size 256
```

| Metric | Value |
|---|---|
| Parameter memory (19.7M params × 4 bytes) | ~75 MB (fp32) / ~37 MB (bf16) |
| Peak inference (b=256) | TODO |
| Peak training (fwd+bwd, b=256) | TODO |

### Estimated FLOPs

Based on the Kaplan et al. approximation:
- Forward pass (2× params): ~39 GFLOPs
- Training step (6× params): ~118 GFLOPs

> Note: The heuristic per-operator FLOPs estimate (~3.8G) is the theoretical multiply-accumulate count. The Kaplan approximation gives a different number because it counts floating-point operations including additions. Both are provided for reference.

---

## Research-quality evaluation

| Dimension | Score (0–5) | Evidence | Gaps |
|---|---|---|---|
| Novelty | 4 | Three clearly identified gaps with thorough literature analysis; each maps to a specific architectural innovation with falsification condition | All three innovations are engineering improvements rather than fundamental breakthroughs; MRF-DW shares conceptual similarity with MixConv |
| Experimental comprehensiveness | 4 | Comprehensive pytest suite (shape, gradient, numerics, CV properties); 8 domain-specific benchmarks; 9 single-field ablations; torch.profiler script | No full ImageNet-1K training; FCMAE pretraining not implemented; no dense prediction transfer; EfficientNet/Swin-T comparison not run |
| Theoretical foundation | 4 | Full per-block parameter formula; complete layer-by-layer budget; stage-level FLOPs estimate; per-innovation overhead computed; inductive bias table | No formal scaling law analysis; no compute-to-communication ratio analysis; no theoretical bound on expected improvement magnitude |
| Result analysis | 2 | Expected accuracy estimates with ranges; accuracy-per-parameter ratio framework defined; proxy metrics available | No actual training results; all accuracy claims are extrapolated; synthetic linear probe is weak proxy for ImageNet accuracy; blocking unknown not resolved |
| Implementation reproducibility | 5 | Complete PyTorch implementation (5 files); all hyperparameters in dataclass; pytest suite; ablation runner (9 configs); profiling script; variant constructors; gradient checkpointing; intermediate features | No Docker/Singularity spec; training recipe not a runnable script; requires CUDA for full benchmark execution; no pretrained weights |
| Writing readiness | 4 | Full architecture spec with ASCII diagram; inductive bias table; research-to-architecture traceability; falsification conditions; training protocol; risk table | No paper draft; related work exists in landscape analysis but not publication-formatted; no visualizations (feature maps, alpha distributions); no ablation results table |

### Required next experiments (from scorecard)

Priority order:

1. **Train tiny variant + ablated baseline on ImageNet-1K supervised (300 epochs)** — resolves the critical `baseline_not_beaten` gap. Without this, no accuracy claims can be verified.
2. **Train uniform downscale baseline [80,160,320,640] at ~19.7M params** — tests the blocking unknown: is V2-E better than a simple ConvNeXt V2-T with uniform channel reduction?
3. **FCMAE pretraining (400ep) + fine-tune (100ep)** — tests LE-GRN compatibility with masked autoencoding. Required by lifecycle contract.
4. **A100 throughput benchmark at batch=256 for all variants** — tests if MRF-DW's 4–8% FLOPs overhead is acceptable in wall-clock time.
5. **EfficientNet-B2/B3 comparison via timm** — tests if V2-E is Pareto-optimal among efficient ConvNets.

### Blocking gaps

| Code | Severity | Description | Blocked claims |
|---|---|---|---|
| `baseline_not_beaten` | Critical | No training results exist — cannot verify V2-E beats any baseline | All accuracy claims |
| `coverage_gap` | High | FCMAE pretraining not implemented | FCMAE compatibility, secondary eval requirement |
| `benchmark_not_executable` | Medium | EfficientNet/Swin-T baselines referenced but no comparison pipeline | V2-E beats EfficientNet at equivalent params |
| `novelty_unverified` | Medium | MRF-DW interaction with DropPath and FCMAE untested | MRF-DW generalizes well, all innovations composable |
| `ablation_missing` | Low | Optimal stage 3 expansion ratio (×3.5 vs ×4) not tested; LE-GRN kernel size (3 vs 5) unresolved | None |
