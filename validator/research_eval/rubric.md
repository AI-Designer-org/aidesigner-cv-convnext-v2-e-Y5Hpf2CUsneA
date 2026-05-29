# ConvNeXt V2-E: Research Quality Evaluation Rubric

## Domain
**Computer Vision (CV)** — Image classification backbone design.

## Task Level
**Level 1** — Idea provided. The user specified: "design improved version of ConvNeXt V2 model with less than 20M parameters." The architecture, innovations (MRF-DW, LE-GRN, Adaptive Expansion), and evaluation protocol were designed by the upstream research and architect stages.

---

## Scoring Dimensions (0–5)

### 1. Novelty
| Score | Criteria |
|-------|----------|
| 0 | No novelty claims made |
| 1 | Claims exist but are vague or well-known (e.g., "deeper model is better") |
| 2 | Claims identify a genuine gap but have partial prior art that covers it |
| 3 | Claims identify gaps with clear differentiation from prior work |
| 4 | Claims are well-motivated, each has a falsification condition, and no single prior work covers them |
| 5 | Claims are demonstrably novel with thorough literature analysis showing the gap — publication-ready |

**CV-specific novelty questions:**
- Does the benchmark suite test multi-scale feature extraction claims (MRF-DW)?
- Does it test normalization effectiveness claims (LE-GRN vs GRN)?
- Does it test stage-capacity allocation claims (adaptive expansion)?
- Does it compare against a budget-matched ConvNeXt V2 baseline?

### 2. Experimental Comprehensiveness
| Score | Criteria |
|-------|----------|
| 0 | No experiments defined |
| 1 | Only forward-pass shape test |
| 2 | Shape + gradient + one domain test |
| 3 | Shape + gradient + numerics + domain benchmarks + ablation framework |
| 4 | Full suite: unit tests, domain benchmarks, ablations, profiling, and research eval |
| 5 | Publication-grade: all of the above + ImageNet-scale training protocol, FCMAE pretraining, dense prediction transfer |

**CV-specific comprehensiveness questions:**
- Does it test translation invariance/equivariance claims?
- Does it test resolution behavior across multiple input sizes?
- Does it measure feature quality (reconstruction probe, linear probe)?
- Does it compare against a simple CNN/ViT-style baseline?
- Are there single-field ablations for every architectural innovation?

### 3. Theoretical Foundation
| Score | Criteria |
|-------|----------|
| 0 | No theoretical justification |
| 1 | Claims mention inspiration but no analysis |
| 2 | Per-block parameter formulas provided |
| 3 | Parameter formulas + FLOPs estimates + inductive bias discussion |
| 4 | Full parameter budget + FLOPs breakdown + scaling reasoning + falsification conditions |
| 5 | Rigorous analysis: compute complexity, memory bandwidth analysis, gradient scaling, ablation bounds |

### 4. Result Analysis
| Score | Criteria |
|-------|----------|
| 0 | No results or result framework |
| 1 | Placeholder for results only |
| 2 | Proxy metrics (noise entropy, gradient norms) reported |
| 3 | Proxy metrics + throughput/memory profiling |
| 4 | Proxy metrics + profiling + expected accuracy estimates with bounds |
| 5 | Full ImageNet results + ablations + statistical significance |

### 5. Implementation Reproducibility
| Score | Criteria |
|-------|----------|
| 0 | No implementation or unusable |
| 1 | Partial implementation with missing components |
| 2 | Complete implementation but no tests |
| 3 | Complete implementation + pytest suite + benchmark scripts |
| 4 | Complete + tested + ablation runner + profiling + documentation |
| 5 | Fully reproducible: configs, seeds, training recipe, pretrained weights, eval scripts |

### 6. Writing Readiness
| Score | Criteria |
|-------|----------|
| 0 | No documentation |
| 1 | Minimal docs (one paragraph) |
| 2 | Architecture description + config docs |
| 3 | Architecture spec + ASCII diagram + inductive bias table + parameter table |
| 4 | Full design doc + research contract + novelty analysis + training protocol |
| 5 | Publication-quality: clear problem statement, related work differentiation, design choices justified, limitations discussed |

---

## Domain-Specific Research Questions (CV)

1. **Multi-scale features**: Does MRF-DW actually learn meaningful per-channel mixing weights, or does α saturate to 0/1 for all channels? (Monitor `alpha` distribution from `torch.sigmoid(mix_weight)` during training.)

2. **Translation invariance**: The model is fully convolutional with global avg pool — how does classification consistency degrade with translation beyond the 7×7 receptive field? The translation robustness sweep in `benchmarks.py` characterizes this.

3. **Resolution robustness**: Since the backbone is fully convolutional (no learned position embeddings), it should handle varying input resolutions. Test at 128, 160, 224, 256, 320, 384.

4. **Feature quality**: The linear probe benchmark provides a quick feature-quality proxy without full ImageNet training. A well-trained backbone should achieve >70% on the synthetic linear probe task.

5. **Parameter efficiency**: The key claim is efficiency under 20M. The parameter efficiency ratio (`expected_acc / param_count_M`) should exceed ConvNeXt V2-T's 2.93%/M.

6. **FCMAE compatibility**: Can LE-GRN handle the zeroed-out regions in masked autoencoding? The local mean subtraction may produce artifacts in masked regions. This remains `TODO: unverified`.

7. **Blocking unknown**: Would a simple uniform downscale of ConvNeXt V2-T (channels [80,160,320,640], uniform ×4) match V2-E's accuracy at the same parameter budget? This is the single most important falsification test.

---

## Evaluation Protocol

The following files constitute the validation suite:

| Artifact | File | What It Tests |
|----------|------|---------------|
| Unit tests | `test_model.py` | Shapes, gradients, numerics, CV properties |
| CV benchmarks | `benchmarks.py` | Reconstruction, linear probe, FLOPs, throughput, memory, noise entropy |
| Ablation runner | `ablation_runner.py` | 9 single-field ablations with proxy eval |
| Profiling | `profile_model.py` | torch.profiler, FLOPs, memory, throughput |
| Scorecard | `scorecard.json` | Machine-readable research quality scores |
| Experiment coverage | `experiment_coverage.md` | Required vs. implemented experiments |
| Claim grounding | `claim_grounding.md` | Every claim mapped to evidence |

### Running the Suite

```bash
# Unit tests
PYTHONPATH=/artifacts/j_Y5Hpf2CUsneA/work/coder:$PYTHONPATH \
    python -m pytest test_model.py -v --tb=short

# Benchmarks
PYTHONPATH=/artifacts/j_Y5Hpf2CUsneA/work/coder:$PYTHONPATH \
    python benchmarks.py

# Ablations
PYTHONPATH=/artifacts/j_Y5Hpf2CUsneA/work/coder:$PYTHONPATH \
    python ablation_runner.py

# Profiling (requires CUDA)
PYTHONPATH=/artifacts/j_Y5Hpf2CUsneA/work/coder:$PYTHONPATH \
    python profile_model.py
```
