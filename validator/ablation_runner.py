"""
ConvNeXt V2-E: Ablation Runner
==============================

Runs each single-field ablation from the architect's specification table.
Every ablation is expressed as a ModelConfig change.

Ablations:
  1. Remove MRF-DW         (use_mrf_dw: True -> False)
  2. Remove LE-GRN          (use_le_grn: True -> False)
  3. Uniform expansion x4   (expansion_ratios: [3,3,4,3] -> [4,4,4,4])
  4. Swap depth distribution (depths: [2,4,10,3] -> [3,3,9,3])
  5. LE-GRN kernel 5x5      (le_grn_local_kernel: 3 -> 5)
  6. MRF-DW small 5x5 early (small_kernel_sizes: [3,3,5,5] -> [5,5,5,5])
  7. Uniform channel downscale (dims: [96,192,384,512] -> [80,160,320,640])
  8. Ablated baseline       (both MRF-DW + LE-GRN off)

Usage:
    PYTHONPATH=/artifacts/j_Y5Hpf2CUsneA/work/coder:$PYTHONPATH python ablation_runner.py
"""

import torch
import torch.nn as nn
import sys
import os
import math

CODER_DIR = os.path.join(os.path.dirname(__file__), "..", "coder")
if CODER_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(CODER_DIR))

from model import ConvNeXtV2E, count_params, count_trainable_params
from model_config import ModelConfig, tiny_config


# ---------------------------------------------------------------------------
# Ablation definitions
# ---------------------------------------------------------------------------

ABLATIONS = {
    "1_full_model": {
        "description": "Full model with all innovations (baseline for comparison)",
        "config": ModelConfig(
            dims=[96, 192, 384, 512],
            depths=[2, 4, 10, 3],
            expansion_ratios=[3.0, 3.0, 4.0, 3.0],
            use_mrf_dw=True,
            small_kernel_sizes=[3, 3, 5, 5],
            use_le_grn=True,
            le_grn_local_kernel=3,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "Full model achieves 80.5-81.5% top-1 supervised",
        "expected_Δparams": 0,
    },
    "2_no_mrf_dw": {
        "description": "Remove MRF-DW (single 7x7 DW) — tests Gap 1",
        "config": ModelConfig(
            dims=[96, 192, 384, 512],
            depths=[2, 4, 10, 3],
            expansion_ratios=[3.0, 3.0, 4.0, 3.0],
            use_mrf_dw=False,
            small_kernel_sizes=[3, 3, 5, 5],
            use_le_grn=True,
            le_grn_local_kernel=3,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "MRF-DW adds >=0.3% top-1 over single 7x7 at same param budget",
        "expected_Δparams": -149376,  # 150K savings
    },
    "3_no_le_grn": {
        "description": "Remove LE-GRN (use standard GRN) — tests Gap 2",
        "config": ModelConfig(
            dims=[96, 192, 384, 512],
            depths=[2, 4, 10, 3],
            expansion_ratios=[3.0, 3.0, 4.0, 3.0],
            use_mrf_dw=True,
            small_kernel_sizes=[3, 3, 5, 5],
            use_le_grn=False,
            le_grn_local_kernel=3,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "LE-GRN adds >=0.2% top-1 over standard GRN at zero param overhead",
        "expected_Δparams": 0,  # Same params as GRN
    },
    "4_uniform_expansion": {
        "description": "Uniform x4 expansion instead of [3,3,4,3] — tests Gap 3",
        "config": ModelConfig(
            dims=[96, 192, 384, 512],
            depths=[2, 4, 10, 3],
            expansion_ratios=[4.0, 4.0, 4.0, 4.0],
            use_mrf_dw=True,
            small_kernel_sizes=[3, 3, 5, 5],
            use_le_grn=True,
            le_grn_local_kernel=3,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "Adaptive expansion saves ~0.95M params with <=0.1% acc loss",
        "expected_Δparams": 952320,  # Approximately +0.95M
    },
    "5_swap_depths": {
        "description": "Swap to [3,3,9,3] depth distribution (V2-T-like)",
        "config": ModelConfig(
            dims=[96, 192, 384, 512],
            depths=[3, 3, 9, 3],
            expansion_ratios=[3.0, 3.0, 4.0, 3.0],
            use_mrf_dw=True,
            small_kernel_sizes=[3, 3, 5, 5],
            use_le_grn=True,
            le_grn_local_kernel=3,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "Shifting capacity S1←S3 improves gradient flow; <=0.1% acc change",
        "expected_Δparams": 0,  # Roughly same total blocks (18 vs 19)
    },
    "6_le_grn_kernel5": {
        "description": "Larger LE-GRN local kernel (5x5 vs 3x3)",
        "config": ModelConfig(
            dims=[96, 192, 384, 512],
            depths=[2, 4, 10, 3],
            expansion_ratios=[3.0, 3.0, 4.0, 3.0],
            use_mrf_dw=True,
            small_kernel_sizes=[3, 3, 5, 5],
            use_le_grn=True,
            le_grn_local_kernel=5,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "Larger local window captures more meaningful context; +0.05% vs 3x3",
        "expected_Δparams": 0,  # Same params, just different avg pool kernel
    },
    "7_mrf_small5_all": {
        "description": "MRF-DW small kernel 5x5 in all stages",
        "config": ModelConfig(
            dims=[96, 192, 384, 512],
            depths=[2, 4, 10, 3],
            expansion_ratios=[3.0, 3.0, 4.0, 3.0],
            use_mrf_dw=True,
            small_kernel_sizes=[5, 5, 5, 5],
            use_le_grn=True,
            le_grn_local_kernel=3,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "Wider small kernel in early stages captures more context; +0.1%",
        "expected_Δparams": 2 * 96 * (25 - 9) + 2 * 192 * (25 - 9),  # S1 + S2 extra
    },
    "8_uniform_downscale": {
        "description": "Uniform channel downscale [80,160,320,640] — strongest baseline",
        "config": ModelConfig(
            dims=[80, 160, 320, 640],
            depths=[3, 3, 9, 3],
            expansion_ratios=[4.0, 4.0, 4.0, 4.0],
            use_mrf_dw=False,
            small_kernel_sizes=[3, 3, 5, 5],
            use_le_grn=False,
            le_grn_local_kernel=3,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "Uniform downscaled V2-T at ~19.7M should be within 0.3% of V2-E",
        "expected_Δparams": 0,  # Should be ~19.7M
    },
    "9_ablated_baseline": {
        "description": "Both innovations off (single DW + standard GRN + adaptive expansion)",
        "config": ModelConfig(
            dims=[96, 192, 384, 512],
            depths=[2, 4, 10, 3],
            expansion_ratios=[3.0, 3.0, 4.0, 3.0],
            use_mrf_dw=False,
            small_kernel_sizes=[3, 3, 5, 5],
            use_le_grn=False,
            le_grn_local_kernel=3,
            drop_path_rate=0.1,
            num_classes=1000,
        ),
        "hypothesis": "Baseline without innovations; isolates combined effect",
        "expected_Δparams": -149376 - 0,  # Remove MRF-DW overhead (~150K)
    },
}


# ---------------------------------------------------------------------------
# Evaluation proxy (synthetic — replace with real training for actual use)
# ---------------------------------------------------------------------------


def evaluate_model_proxy(model: nn.Module) -> dict:
    """
    Proxy evaluation: compute metrics that don't require full training.

    In a real research setting, this would train on ImageNet-1K for 300 epochs.
    Here we compute:
      - Forward pass sanity (no NaN, correct shape)
      - Backward pass sanity (gradients flow)
      - Noise entropy (proxy for miscalibration)
      - Gradient norm statistics (proxy for trainability)
    """
    device = next(model.parameters()).device
    model.train()

    results = {
        "forward_ok": False,
        "backward_ok": False,
        "noise_entropy": 0.0,
        "mean_grad_norm": 0.0,
        "zero_grad_params": [],
        "nan_grad_params": [],
        "grad_flow": {},
    }

    # Forward
    try:
        x = torch.randn(4, 3, 224, 224, device=device)
        logits = model(x)
        assert logits.shape == (4, model.cfg.num_classes)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
        results["forward_ok"] = True
    except Exception as e:
        results["forward_error"] = str(e)
        return results

    # Noise entropy (eval mode)
    model.eval()
    with torch.no_grad():
        x_noise = torch.randn(32, 3, 224, 224, device=device)
        logits_noise = model(x_noise)
        probs = logits_noise.softmax(-1)
        entropy = -(probs * probs.log()).sum(-1).mean().item()
        max_entropy = math.log(model.cfg.num_classes)
        results["noise_entropy"] = entropy
        results["noise_entropy_fraction"] = entropy / max_entropy

    # Backward pass + gradient stats
    model.train()
    x = torch.randn(4, 3, 224, 224, device=device, requires_grad=True)
    target = torch.randint(0, model.cfg.num_classes, (4,), device=device)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()

    results["backward_ok"] = True
    results["loss"] = loss.item()

    total_norm = 0.0
    zero_grad = []
    nan_grad = []
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            param_norm = p.grad.norm().item()
            total_norm += param_norm
            if param_norm < 1e-10:
                zero_grad.append(name)
            if torch.isnan(p.grad).any():
                nan_grad.append(name)
            # Store per-component grad norms
            parts = name.split(".")
            component = parts[0] if len(parts) > 0 else "root"
            if component not in results["grad_flow"]:
                results["grad_flow"][component] = []
            results["grad_flow"][component].append(param_norm)

    results["mean_grad_norm"] = total_norm / max(
        1, sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    )
    results["zero_grad_params"] = zero_grad
    results["nan_grad_params"] = nan_grad

    # Summarize per-component grad norms
    for comp, norms in results["grad_flow"].items():
        results["grad_flow"][comp] = {
            "mean": sum(norms) / len(norms),
            "max": max(norms),
            "min": min(norms),
        }

    return results


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_ablations():
    """Run all ablations and print summary."""
    print("=" * 80)
    print("ConvNeXt V2-E: Ablation Runner")
    print("=" * 80)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    results = {}
    baselines = {}

    for ablation_id, spec in ABLATIONS.items():
        print("-" * 80)
        print(f"Ablation: {ablation_id}")
        print(f"  Description: {spec['description']}")
        print(f"  Hypothesis: {spec['hypothesis']}")
        print(f"  Expected Δparams: {spec['expected_Δparams']:+,}")
        print()

        cfg = spec["config"]
        model = ConvNeXtV2E(cfg).to(device)
        total_params = count_params(model)
        trainable_params = count_trainable_params(model)
        within_budget = total_params < 20_000_000

        print(f"  Total params:     {total_params:>10,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable params: {trainable_params:>10,}")
        print(f"  Within 20M budget: {within_budget}")

        # Parameter delta vs full model
        if "1_full_model" in results:
            delta = total_params - results["1_full_model"]["params"]
            print(f"  Δparams vs full:  {delta:>+10,} ({delta/results['1_full_model']['params']*100:+.2f}%)")
        else:
            # First pass — save for later comparison
            delta = 0

        print()

        # Run proxy evaluation
        print("  Running evaluation proxy...")
        eval_results = evaluate_model_proxy(model)

        if eval_results["forward_ok"]:
            print(f"  [OK] Forward pass")
        else:
            print(f"  [FAIL] Forward pass: {eval_results.get('forward_error', 'unknown')}")

        if eval_results["backward_ok"]:
            print(f"  [OK] Backward pass (loss: {eval_results['loss']:.4f})")
            print(f"  [OK] Mean grad norm: {eval_results['mean_grad_norm']:.4e}")
            if eval_results["zero_grad_params"]:
                print(f"  [WARN] Zero-grad params: {eval_results['zero_grad_params']}")
            if eval_results["nan_grad_params"]:
                print(f"  [FAIL] NaN-grad params: {eval_results['nan_grad_params']}")
        else:
            print(f"  [FAIL] Backward pass")

        print(f"  Noise entropy: {eval_results['noise_entropy']:.3f} "
              f"({eval_results['noise_entropy_fraction']:.1%} of max)")

        print()

        # Save
        results[ablation_id] = {
            "params": total_params,
            "trainable": trainable_params,
            "within_budget": within_budget,
            "delta_params_vs_full": delta,
            "eval": eval_results,
        }

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Ablation':<30s} {'Params':>10s} {'Budget':>8s} {'Forward':>8s} {'Backward':>8s} {'Entropy':>8s}")
    print("-" * 80)

    for ablation_id in sorted(results.keys()):
        r = results[ablation_id]
        p = r["params"]
        budget = "OK" if r["within_budget"] else "OVER"
        fwd = "OK" if r["eval"]["forward_ok"] else "FAIL"
        bwd = "OK" if r["eval"]["backward_ok"] else "FAIL"
        ent = f"{r['eval']['noise_entropy']:.2f}"
        print(f"{ablation_id:<30s} {p:>10,} {budget:>8s} {fwd:>8s} {bwd:>8s} {ent:>8s}")

    print()
    print("=" * 80)

    # Compute parameter deltas for key comparisons
    if "1_full_model" in results and "2_no_mrf_dw" in results:
        mrf_delta = results["2_no_mrf_dw"]["params"] - results["1_full_model"]["params"]
        print(f"MRF-DW overhead:        {mrf_delta:+,} params")
    if "1_full_model" in results and "3_no_le_grn" in results:
        le_grn_delta = results["3_no_le_grn"]["params"] - results["1_full_model"]["params"]
        print(f"LE-GRN overhead:         {le_grn_delta:+,} params")
    if "4_uniform_expansion" in results and "1_full_model" in results:
        exp_delta = results["4_uniform_expansion"]["params"] - results["1_full_model"]["params"]
        print(f"Adaptive expansion saves: {exp_delta:+,} params")
    if "9_ablated_baseline" in results and "1_full_model" in results:
        innovation_delta = results["9_ablated_baseline"]["params"] - results["1_full_model"]["params"]
        print(f"Innovation overhead:    {innovation_delta:+,} params "
              f"({100*innovation_delta/results['1_full_model']['params']:.2f}%)")

    return results


if __name__ == "__main__":
    run_ablations()
