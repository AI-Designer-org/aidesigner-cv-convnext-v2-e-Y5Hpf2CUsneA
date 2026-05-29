# Changelog

## [0.1.0] — 2026-05-29

### Added
- Initial implementation of ConvNeXt V2-E, an efficient ConvNet backbone under 20M parameters.
- Three architectural innovations over ConvNeXt V2:
  - **MRF-DW**: Multi-Receptive Field Depthwise Convolution with learnable per-channel mixing (<0.8% parameter overhead)
  - **LE-GRN**: Local-Enhanced Global Response Normalization at zero additional parameter cost
  - **Adaptive Expansion**: Stage-specific expansion ratios [3,3,4,3] saving ~1.9M params vs uniform ×4
- Implementation files:
  - `model_config.py` — ModelConfig and TrainingConfig dataclasses with variant constructors
  - `layers.py` — LayerNormNd, DropPath, Stem, DownsampleBlock, ConvNeXtV2EBlock
  - `mrf_dw.py` — MRFDWConv, StandardDWConv, and factory function
  - `le_grn.py` — LEGRN, GRN, and factory function
  - `model.py` — ConvNeXtV2E backbone, variant constructors, parameter helpers
- Comprehensive test suite (pytest, 10-test smoke suite):
  - Shape tests (standard forward, batch=1/16, variable resolution 128–384)
  - Gradient flow tests (all params receive gradients, no NaN grads, MRF-DW/LE-GRN param grads)
  - CV-specific correctness (translation invariance, noise entropy, batch independence)
  - Numerical stability (fp16/bf16 forward, bf16 backward, extreme inputs ±1000, zero input)
  - Innovation-specific tests (MRF-DW mixing weight shape/init/range, LE-GRN vs GRN diff/param parity)
  - DropPath correctness, weight initialization, block-level tests, config validation
- Domain-specific CV benchmarks:
  - Masked patch reconstruction probe
  - Linear probe on synthetic classification
  - Parameter efficiency ratio (accuracy-per-param)
  - Noise entropy profile
  - Translation robustness sweep
  - FLOPs estimation (heuristic + fvcore)
  - Throughput measurement (infrastructure)
  - Memory footprint (infrastructure)
- Ablation runner with 9 single-field ablation configurations covering every innovation
- Profiling script: torch.profiler for forward/train-step, throughput, memory, FLOPs
- Research evaluation artifacts: scorecard, experiment coverage, claim grounding
- Documentation: README, ARCHITECTURE, TRAINING, BENCHMARKS, API, CHANGELOG
