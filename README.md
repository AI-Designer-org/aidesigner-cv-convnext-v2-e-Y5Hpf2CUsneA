> **Project layout** — this bundle contains five stage directories from the
> AI-Designer pipeline:
> `research/` (literature survey), `architect/` (blueprint + `ModelConfig`),
> `coder/` (PyTorch implementation), `validator/` (tests + benchmarks), and
> `documenter/` (this README plus `docs/` and `CHANGELOG.md`).
> An optional `paper/` directory holds the NeurIPS-format writeup when the
> paper-generation step was triggered.
>
> The original research request that produced this bundle is preserved
> verbatim in [`prompt.md`](prompt.md) — if any URLs in the prompt were
> fetched server-side for additional context, their cleaned contents are
> appended there too.

---

# ConvNeXt V2-E

A redesigned ConvNeXt V2 variant under 20M parameters with multi-scale depthwise mixing, local-enhanced feature competition, and stage-adaptive expansion ratios.

ConvNeXt V2-E addresses the gap between parameter-efficient ConvNets (EfficientNet, MobileNet) and modernized ConvNeXt-style designs at the sub-20M regime. The architecture introduces three lightweight innovations — Multi-Receptive Field Depthwise Convolution (MRF-DW), Local-Enhanced Global Response Normalization (LE-GRN), and stage-adaptive expansion ratios — that collectively improve representational efficiency at a 19.7M parameter budget (70% of ConvNeXt V2-T). Estimated FLOPs are ~3.8G, a ~16% reduction from V2-T's ~4.5G.

> **Note**: All accuracy claims below are extrapolated from the ConvNeXt V2 scaling curve and are marked `TODO: unverified` pending ImageNet-1K training. See [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md#research-quality-evaluation) for the current verification status.

## Highlights

- **Multi-scale spatial mixing per block** — MRF-DW runs parallel 7×7 and 3×3/5×5 depthwise convolutions with learnable per-channel mixing, at <0.8% parameter overhead. See [ARCHITECTURE.md#3-the-core-component](documenter/docs/ARCHITECTURE.md#3-the-core-component).
- **Local-enhanced feature competition at zero parameter cost** — LE-GRN augments GRN with local mean subtraction (3×3 avg pool) before global RMS normalization, using the same 2C parameters as standard GRN. See [ARCHITECTURE.md#32-innovation-2-le-grn](documenter/docs/ARCHITECTURE.md#32-innovation-2-le-grn).
- **Stage-adaptive expansion ratios** — [3,3,4,3] vs uniform ×4 saves ~1.9M parameters (~10% of total) by allocating more capacity to Stage 3 (10 blocks × e=4) and less to high-resolution early stages. See [ARCHITECTURE.md#33-innovation-3-adaptive-expansion](documenter/docs/ARCHITECTURE.md#33-innovation-3-adaptive-expansion).
- **Verified under 20M budget** — All variants (tiny/wide/deep/ablated) pass parameter budget checks; confirmed by smoke test and pytest suite.

## Quick start

```bash
cd /artifacts/j_Y5Hpf2CUsneA/work/coder
pip install -r requirements.txt  # requires torch, torchvision
python smoke_test.py             # 10-test smoke suite
cd /artifacts/j_Y5Hpf2CUsneA/work/validator
PYTHONPATH=../coder:$PYTHONPATH python -m pytest test_model.py -v --tb=short
```

## Repository layout

The implementation lives in the coder stage directory. Documentation is in this directory.

```
coder/
  model.py          # ConvNeXtV2E backbone, variant constructors, param helpers
  model_config.py   # ModelConfig and TrainingConfig dataclasses
  layers.py         # LayerNormNd, DropPath, Stem, DownsampleBlock, ConvNeXtV2EBlock
  mrf_dw.py         # MRFDWConv, StandardDWConv, build_spatial_mixer
  le_grn.py         # LEGRN, GRN, build_feature_competition
  smoke_test.py     # 10-test smoke verification suite
validator/
  test_model.py     # Comprehensive pytest suite (shapes, gradients, CV, numerics)
  benchmarks.py     # Domain-specific CV benchmarks
  ablation_runner.py # 9 single-field ablation configurations
  profile_model.py  # FLOPs, memory, throughput profiling
  research_eval/    # Scorecard, experiment coverage, claim grounding
documenter/
  README.md         # This file
  docs/
    ARCHITECTURE.md # Design, inductive biases, equations, shape evolution
    TRAINING.md     # Training recipe and hyperparameters
    BENCHMARKS.md   # Benchmark results and profiling
    API.md          # Module-level API reference
CHANGELOG.md        # Version history
```

## Documentation

- [docs/ARCHITECTURE.md](documenter/docs/ARCHITECTURE.md) — design, innovations, inductive biases, shape evolution
- [docs/TRAINING.md](documenter/docs/TRAINING.md) — supervised training recipe, hyperparameter rationale, troubleshooting
- [docs/BENCHMARKS.md](documenter/docs/BENCHMARKS.md) — synthetic benchmarks, ablation study, profiling, research-quality evaluation
- [docs/API.md](documenter/docs/API.md) — module-level API reference with shape contracts

## Citation

```bibtex
@misc{convnext-v2e,
  title  = {ConvNeXt V2-E: Efficient ConvNet under 20M Parameters with Multi-Scale Depthwise Mixing and Local-Enhanced Normalization},
  author = {TODO: author name},
  year   = {2026},
  note   = {Generated via ml-designer pipeline}
}
```
