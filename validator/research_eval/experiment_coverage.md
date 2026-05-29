# ConvNeXt V2-E: Experiment Coverage

## Required vs. Implemented Experiments

This file maps the lifecycle contract requirements from `research/03_novelty_gaps_and_lifecycle_contract.yaml` to the actual validation artifacts.

---

## Evaluation Requirements (from Lifecycle Contract)

### Primary: Top-1 accuracy on ImageNet-1K validation (224×224), supervised 300 epochs
| Status | Artifact | Notes |
|--------|----------|-------|
| ❌ Not implemented | — | Requires a full training pipeline (distributed training on ImageNet-1K). The validator provides proxy metrics only. |
| ◐ Proxy available | `benchmarks.py::linear_probe_benchmark` | Synthetic linear probe on frozen features — not a substitute for true ImageNet accuracy |

### Secondary: Top-1 with FCMAE pretraining (400ep + 100ep fine-tune)
| Status | Artifact | Notes |
|--------|----------|-------|
| ❌ Not implemented | — | Requires FCMAE pretraining framework with sparse conv decoder |

### Ablation: Top-1 with each innovation ablated
| Status | Artifact | Notes |
|--------|----------|-------|
| ✅ Ablation framework | `ablation_runner.py` | 9 single-field ablations, each with proxy evaluation |
| ❌ Actual accuracy | — | Requires full training runs for each ablation |
| ✅ Per-innovation param counts | `ablation_runner.py` summary table | Verified MRF-DW ~150K overhead, LE-GRN 0 overhead, adaptive expansion ~0.95M savings |

### Efficiency: Throughput on single A100 at batch=256
| Status | Artifact | Notes |
|--------|----------|-------|
| ✅ Script | `profile_model.py::measure_throughput` | Measures throughput at multiple batch sizes |
| ⚠️ Requires CUDA | — | Results depend on GPU hardware |

### Memory: Peak GPU memory at batch=256
| Status | Artifact | Notes |
|--------|----------|-------|
| ✅ Script | `profile_model.py::profile_memory` | Reports param memory, inference peak, training peak |
| ⚠️ Requires CUDA | — | Results depend on GPU hardware |

### Compute: Total FLOPs per forward pass (224×224)
| Status | Artifact | Notes |
|--------|----------|-------|
| ✅ fvcore + heuristic | `profile_model.py::flop_count`, `profile_model.py::estimate_flops_formula` | Both methods available; fvcore preferred |
| ✅ Per-stage breakdown | `architect/02_architecture_spec.md` | Detailed stage-level FLOPs table |

---

## Baseline Requirements

| Baseline | Params | Status | Implementation |
|----------|--------|--------|----------------|
| ConvNeXt V2-T (28M) | 28M | ✅ Published result referenced | `benchmarks.py::parameter_efficiency_ratio` |
| V2-E ablated baseline (~19.5M) | ~19.5M | ✅ Implemented | `convnext_v2e_ablated_baseline()` in `model.py` |
| Uniform downscale V2 (~19.7M) | ~19.7M | ✅ Implemented | `ablation_runner.py` ablation 8 |
| EfficientNet-B2 | 9.2M | ❌ Not implemented | Requires external model |
| EfficientNet-B3 | 12M | ❌ Not implemented | Requires external model |
| Swin-T | 28M | ❌ Not implemented | Requires external model |
| RegNetY-4GF | 21M | ❌ Not implemented | Requires external model |

---

## Ablation Coverage

From the architect's suggested ablation table (`architect/02_architecture_spec.md`):

| # | Ablation | Config Change | Implemented | In |
|---|----------|--------------|-------------|-----|
| 1 | Remove MRF-DW | `use_mrf_dw: True→False` | ✅ | `ablation_runner.py` #2 |
| 2 | Remove LE-GRN | `use_le_grn: True→False` | ✅ | `ablation_runner.py` #3 |
| 3 | Uniform expansion ×4 | `expansion_ratios: [3,3,4,3]→[4,4,4,4]` | ✅ | `ablation_runner.py` #4 |
| 4 | Swap depth distribution | `depths: [2,4,10,3]→[3,3,9,3]` | ✅ | `ablation_runner.py` #5 |
| 5 | LE-GRN kernel 5×5 | `le_grn_local_kernel: 3→5` | ✅ | `ablation_runner.py` #6 |
| 6 | MRF-DW small 5×5 in S1-2 | `small_kernel_sizes: [3,3,5,5]→[5,5,5,5]` | ✅ | `ablation_runner.py` #7 |
| 7 | FCMAE pretraining | (training config) | ❌ | Requires training pipeline |
| 8 | Uniform channel downscale | `dims: [96,192,384,512]→[80,160,320,640]` | ✅ | `ablation_runner.py` #8 |

**Additionally implemented:**
- Combined ablated baseline (both MRF-DW + LE-GRN off) — `ablation_runner.py` #9

---

## Synthetic Benchmarks Implemented

| Benchmark | File | What It Measures |
|-----------|------|-----------------|
| Masked patch reconstruction probe | `benchmarks.py` | MSE of encoder-decoder reconstruction vs random baseline |
| Linear probe (CIFAR-like synthetic) | `benchmarks.py` | Frozen feature quality on synthetic classification |
| Parameter efficiency ratio | `benchmarks.py` | Expected accuracy per million params |
| Noise entropy profile | `benchmarks.py` | Prediction entropy on pure noise (calibration proxy) |
| Translation robustness sweep | `benchmarks.py` | Classification consistency vs pixel shift |
| FLOPs estimation | `benchmarks.py`, `profile_model.py` | GFLOPs via fvcore and heuristic |
| Throughput measurement | `benchmarks.py`, `profile_model.py` | images/sec at multiple batch sizes |
| Memory footprint | `benchmarks.py`, `profile_model.py` | Parameter, inference, and training peak memory |

---

## Metrics Reported

| Metric | Available? | How |
|--------|-----------|-----|
| Total parameters | ✅ | `count_params()` in `model.py` — all variants verified < 20M |
| Trainable parameters | ✅ | `count_trainable_params()` |
| Per-stage parameter distribution | ✅ | `print_model_summary()` |
| Per-innovation parameter overhead | ✅ | `smoke_test.py`, `ablation_runner.py` |
| Forward pass health (shape, NaN) | ✅ | `test_model.py::TestShapes`, `test_model.py::TestNumerics` |
| Gradient health (flow, NaN, norms) | ✅ | `test_model.py::TestGradients`, `ablation_runner.py::evaluate_model_proxy` |
| fp16/bf16 numerical stability | ✅ | `test_model.py::TestNumerics` |
| Noise entropy | ✅ | `benchmarks.py::noise_entropy_profile` |
| Translation robustness | ✅ | `benchmarks.py::translation_robustness_sweep` |
| FLOPs | ✅ | `profile_model.py` (requires CUDA for precise) |
| Throughput | ✅ | `profile_model.py` (requires CUDA) |
| Memory | ✅ | `profile_model.py` (requires CUDA) |

**Not yet available (TODO: unverified):**
- Top-1 accuracy on ImageNet-1K
- Top-1 accuracy with FCMAE
- Per-ablation accuracy deltas
- Wall-clock training time
- Comparison with external baselines (EfficientNet, Swin-T)

---

## Can the Benchmark Suite Distinguish V2-E from a Trivial Baseline?

**Partially.** The suite can verify:

1. **Parameter budget compliance**: ✅ Yes — assert all variants < 20M
2. **Gradient health**: ✅ Yes — assert all params receive finite gradients
3. **Numerical stability**: ✅ Yes — assert no NaN/Inf in fp16/bf16
4. **Translation robustness**: ✅ Yes — classification consistency vs shift
5. **Memory-bound overhead of MRF-DW**: ✅ Yes — throughput comparison of full vs. ablated
6. **Zero-param cost of LE-GRN**: ✅ Yes — verified same param count as GRN
7. **Adaptive expansion savings**: ✅ Yes — verified ~0.95M params saved vs uniform ×4

**But cannot distinguish on accuracy** without full training. The synthetic linear probe and noise entropy are weak proxies. A trivial last-layer-classifier baseline could match the linear probe with random features.

**Recommendation**: The highest-impact next step is to train the tiny variant and the ablated baseline on ImageNet-1K. Without those results, the accuracy claims remain unvalidated hypotheses.
