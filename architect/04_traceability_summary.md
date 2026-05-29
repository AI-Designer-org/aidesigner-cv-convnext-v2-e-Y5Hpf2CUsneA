# ConvNeXt V2-E: Research-to-Architecture Traceability Summary

## Innovation-to-Implementation Mapping

| Innovation | Files | Key Hyperparameters | Validation |
|---|---|---|---|
| **MRF-DW** | `03_model_implementation.py:MRFDWConv` | `use_mrf_dw`, `small_kernel_sizes`, `mrf_mix_init`, `base_kernel_size` | Ablate `use_mrf_dw=False` |
| **LE-GRN** | `03_model_implementation.py:LEGRN` | `use_le_grn`, `le_grn_local_kernel`, `grn_eps` | Ablate `use_le_grn=False` |
| **Adaptive Expansion** | `01_model_config.py:ModelConfig` | `expansion_ratios=[3,3,4,3]`, `depths=[2,4,10,3]` | Compare vs `expansion_ratios=[4,4,4,4]`, `depths=[3,3,9,3]` |

---

## File Manifest

| File | Purpose |
|---|---|
| `01_model_config.py` | ModelConfig + TrainingConfig dataclasses; variant constructors |
| `02_architecture_spec.md` | Full spec: pseudocode, ASCII diagrams, justification, ablations, risks |
| `03_model_implementation.py` | Run-ready PyTorch implementation of ConvNeXtV2E |
| `04_traceability_summary.md` | This file — index and cross-reference |

---

## Lifecycle Contract Handoff

| Research Contract Item | Delivered | Artifact |
|---|---|---|
| task_level: level_1 | ModelConfig + implementation for 224×224 IN-1K | `01_model_config.py`, `03_model_implementation.py` |
| novelty_claims | Each claim encoded as a config toggle | `01_model_config.py` fields `use_mrf_dw`, `use_le_grn`, `expansion_ratios` |
| baseline_requirements | V2-T budget-matched ablated baseline variant | `convnext_v2e_ablated_baseline()` in `03_model_implementation.py` |
| evaluation_requirements | Model supports throughput, memory, FLOPs measurement | `forward()` + `get_intermediate_features()` in `03_model_implementation.py` |
| falsification_conditions | Each ablation maps to a single config-field change | Ablation table in `02_architecture_spec.md` |
| blocking_unknowns | Flagged as `TODO: unverified` in traceability table | `02_architecture_spec.md` Research-to-Architecture Traceability section |

---

## Expected Accuracy Estimates (Hypotheses)

| Setting | Expected Top-1 (IN-1K, 224×224, 300ep supervised) | Source |
|---|---|---|
| ConvNeXt V2-T (28M) — supervised | ~82.0% | Published (V2-T with FCMAE = 82.8%; supervised is ~0.8% lower) |
| V2-E full (19.7M, all innovations) | **80.5–81.5%** | Extrapolated from V2-T scaling curve |
| V2-E ablated base (no MRF-DW, no LE-GRN) | 80.2–81.0% | Baseline regression |
| V2-E + FCMAE pretraining | **81.5–82.5%** | FCMAE typically adds ~1.0% |
| V2-E uniform ×4 (20.5M, over budget) | 80.5–81.5% | Same accuracy, 0.95M more params |

**Key comparison**: If V2-E achieves ≥81.5% supervised, it beats the 82.8% V2-T (FCMAE) accuracy-per-parameter ratio:
- V2-T: 82.8% / 28M = 2.96% / M
- V2-E: 81.5% / 19.7M = 4.14% / M  → **40% better efficiency**

Even at 80.5% supervised: 80.5 / 19.7 = 4.09% / M → **38% better efficiency**.

---

## Falsification Conditions (Short Form)

1. **Model-level**: V2-E < 80.0% top-1 (supervised, 300ep, 224×224) → full design invalidated
2. **MRF-DW**: Δ < 0.15% top-1 when ablated → Gap 1 research claim falsified
3. **LE-GRN**: Δ < 0.15% top-1 when ablated → Gap 2 research claim falsified
4. **Adaptive expansion**: Uniform ×4 > 0.3% better at same budget → Gap 3 allocation strategy wrong
5. **Uniform downscale baseline**: [80,160,320,640] uniform V2 ≤ 0.3% behind V2-E → innovations not worth complexity
