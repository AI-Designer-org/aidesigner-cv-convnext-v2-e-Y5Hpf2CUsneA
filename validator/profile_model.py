"""
ConvNeXt V2-E: Profiling Script
===============================

Profiles the ConvNeXt V2-E model for:
  - Forward pass FLOPs, memory, and timing (torch.profiler)
  - Forward+backward training step
  - Per-operator breakdown
  - Peak memory usage
  - Throughput estimation

Usage:
    PYTHONPATH=/artifacts/j_Y5Hpf2CUsneA/work/coder:$PYTHONPATH python profile_model.py
"""

import torch
import torch.nn as nn
import sys
import os
import time
import argparse

CODER_DIR = os.path.join(os.path.dirname(__file__), "..", "coder")
if CODER_DIR not in sys.path:
    sys.path.insert(0, os.path.abspath(CODER_DIR))

from model import ConvNeXtV2E, count_params
from model_config import tiny_config, ModelConfig


# ---------------------------------------------------------------------------
# FLOPs estimation with fvcore (preferred) or heuristic fallback
# ---------------------------------------------------------------------------


def flop_count(model: nn.Module, input_size=(1, 3, 224, 224)) -> float:
    """Count FLOPs using fvcore if available, else heuristic."""
    try:
        from fvcore.nn import FlopCountAnalysis
        x = torch.randn(input_size).to(next(model.parameters()).device)
        flops = FlopCountAnalysis(model, x)
        total = flops.total() / 1e9
        print(f"  [fvcore] Total FLOPs: {total:.2f} G")
        return total
    except ImportError:
        print("  [fvcore] Not installed. Use: pip install fvcore")
        return 0.0


# ---------------------------------------------------------------------------
# torch.profiler profiling
# ---------------------------------------------------------------------------


def profile_forward(
    model: nn.Module,
    batch_size: int = 32,
    steps: int = 5,
    warmup: int = 3,
):
    """Profile forward pass with torch.profiler."""
    if not torch.cuda.is_available():
        print("[Profile] CUDA required for profiling. Skipping.")
        return

    from torch.profiler import profile, record_function, ProfilerActivity

    device = next(model.parameters()).device
    model.eval()

    x = torch.randn(batch_size, 3, 224, 224, device=device)

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize()

    print(f"\n{'='*70}")
    print(f"Profiler: Forward pass (batch={batch_size}, {steps} steps)")
    print(f"{'='*70}")

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            with record_function("forward"):
                with torch.no_grad():
                    out = model(x)
            torch.cuda.synchronize()

    print("\n--- Top 15 by CUDA time ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n--- Top 15 by CUDA memory ---")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))

    return prof


def profile_train_step(
    model: nn.Module,
    batch_size: int = 32,
    steps: int = 5,
    warmup: int = 3,
):
    """Profile forward+backward training step with torch.profiler."""
    if not torch.cuda.is_available():
        print("[Profile] CUDA required for profiling. Skipping.")
        return

    from torch.profiler import profile, record_function, ProfilerActivity

    device = next(model.parameters()).device
    model.train()

    # Warmup
    for _ in range(warmup):
        x = torch.randn(batch_size, 3, 224, 224, device=device)
        target = torch.randint(0, model.cfg.num_classes, (batch_size,), device=device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, target)
        loss.backward()
    torch.cuda.synchronize()

    # Reset gradients after warmup
    model.zero_grad()

    print(f"\n{'='*70}")
    print(f"Profiler: Train step (fwd+bwd, batch={batch_size}, {steps} steps)")
    print(f"{'='*70}")

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(steps):
            x = torch.randn(batch_size, 3, 224, 224, device=device)
            target = torch.randint(0, model.cfg.num_classes, (batch_size,), device=device)

            with record_function("train_step"):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, target)
                loss.backward()
            torch.cuda.synchronize()

    print("\n--- Top 15 by CUDA time (train step) ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    print("\n--- Top 15 by self CUDA memory (train step) ---")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=15))

    return prof


# ---------------------------------------------------------------------------
# Throughput measurement
# ---------------------------------------------------------------------------


def measure_throughput(
    model: nn.Module,
    batch_sizes: list = None,
    n_warmup: int = 20,
    n_iters: int = 100,
):
    """Measure inference throughput at multiple batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64, 128, 256]

    if not torch.cuda.is_available():
        print("[Throughput] CUDA required. Skipping.")
        return

    device = next(model.parameters()).device
    model.eval()

    print(f"\n{'='*70}")
    print("Throughput Measurement")
    print(f"{'='*70}")
    print(f"{'Batch':>8s} {'Images/s':>12s} {'ms/img':>10s} {'Memory':>12s}")
    print("-" * 70)

    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 3, 224, 224, device=device)

        # Warmup
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()

        # Clear peak memory stats
        torch.cuda.reset_peak_memory_stats()

        # Timed runs
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = model(x)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        total_images = bs * n_iters
        imgs_per_sec = total_images / elapsed
        ms_per_img = 1000.0 / imgs_per_sec
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024

        results[bs] = {
            "images_per_sec": imgs_per_sec,
            "ms_per_image": ms_per_img,
            "peak_memory_mb": peak_mem,
        }

        print(f"{bs:>8d} {imgs_per_sec:>12.0f} {ms_per_img:>10.3f} {peak_mem:>10.1f} MB")

    return results


# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------


def profile_memory(
    model: nn.Module,
    batch_size: int = 256,
):
    """Detailed memory breakdown."""
    if not torch.cuda.is_available():
        print("[Memory] CUDA required. Skipping.")
        return

    device = next(model.parameters()).device
    model.eval()

    # Parameter memory
    param_mem = sum(p.numel() * p.element_size() for p in model.parameters())
    param_mem_mb = param_mem / 1024 / 1024

    # Buffer memory (if any)
    buf_mem = sum(b.numel() * b.element_size() for b in model.buffers())
    buf_mem_mb = buf_mem / 1024 / 1024

    # Peak inference memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    x = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        _ = model(x)
    torch.cuda.synchronize()

    peak_inference = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 / 1024

    # Peak training memory (fwd + bwd)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch_size, 3, 224, 224, device=device, requires_grad=True)
    target = torch.randint(0, model.cfg.num_classes, (batch_size,), device=device)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()
    torch.cuda.synchronize()

    peak_training = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    peak_training_reserved = torch.cuda.max_memory_reserved(device) / 1024 / 1024

    print(f"\n{'='*70}")
    print("Memory Profile")
    print(f"{'='*70}")
    print(f"  Parameter memory:              {param_mem_mb:>8.1f} MB")
    print(f"  Buffer memory:                 {buf_mem_mb:>8.1f} MB")
    print(f"  Peak inference (b={batch_size}): {peak_inference:>8.1f} MB")
    print(f"  Peak reserved (inference):     {peak_reserved:>8.1f} MB")
    print(f"  Peak training (b={batch_size}):  {peak_training:>8.1f} MB")
    print(f"  Peak reserved (training):      {peak_training_reserved:>8.1f} MB")

    return {
        "param_mb": param_mem_mb,
        "buffer_mb": buf_mem_mb,
        "inference_peak_mb": peak_inference,
        "training_peak_mb": peak_training,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary(model: nn.Module):
    """Print model summary."""
    total = count_params(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*70}")
    print("ConvNeXt V2-E Summary")
    print(f"{'='*70}")
    print(f"  Total params:     {total:>10,} ({total/1e6:.2f}M)")
    print(f"  Trainable params: {trainable:>10,} ({trainable/1e6:.2f}M)")
    print(f"  Within 20M:       {total < 20_000_000}")
    print(f"  Config:")
    print(f"    dims:              {model.cfg.dims}")
    print(f"    depths:            {model.cfg.depths}")
    print(f"    expansion_ratios:  {model.cfg.expansion_ratios}")
    print(f"    use_mrf_dw:        {model.cfg.use_mrf_dw}")
    print(f"    use_le_grn:        {model.cfg.use_le_grn}")
    print(f"    small_kernel:      {model.cfg.small_kernel_sizes}")


N_QUERIES_PARAMS = 6


def estimate_flops_formula(model: nn.Module) -> float:
    """
    Compute an approximate FLOPs count by reasoning over the architecture.
    Used as a cross-check against fvcore.

    Returns GFLOPs.
    """
    dims = model.cfg.dims
    depths = model.cfg.depths
    exp_ratios = model.cfg.expansion_ratios
    small_ks = model.cfg.small_kernel_sizes

    total = 0.0

    # Stem: 4x4 conv stride 4: input 3x224x224 -> 96x56x56
    total += 2 * 3 * 96 * 4 * 4 * 56 * 56  # MACs * 2

    hw = 56
    for i, (d, dim, exp, sk) in enumerate(zip(depths, dims, exp_ratios, small_ks)):
        n = hw * hw
        for _ in range(d):
            # Base DW 7x7: 2 * n * dim * 49
            total += 2 * n * dim * 49
            # Small DW: 2 * n * dim * sk^2
            total += 2 * n * dim * (sk ** 2)
            # PW1 expand: 2 * n * dim * dim * exp
            total += 2 * n * dim * dim * exp
            # PW2 project: 2 * n * dim * dim * exp
            total += 2 * n * dim * dim * exp
            # GRN/LE-GRN: ~ 2 * n * dim (element-wise)
            total += 2 * n * dim

        # Downsampling: LN + 2x2 conv stride 2
        if i < len(depths) - 1:
            next_dim = dims[i + 1]
            total += 2 * n * dim * next_dim * 4  # 2x2 conv
            hw //= 2

    # Head: FC dim*num_classes + bias
    total += 2 * dims[-1] * model.cfg.num_classes

    return total / 1e9


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Profile ConvNeXt V2-E")
    parser.add_argument("--variant", type=str, default="tiny",
                        choices=["tiny", "wide", "deep", "ablated"])
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for profiling")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of profiler steps")
    parser.add_argument("--no-profile", action="store_true",
                        help="Skip torch.profiler (just timing + memory)")
    args = parser.parse_args()

    # Build model
    from model import (
        convnext_v2e_tiny, convnext_v2e_wide,
        convnext_v2e_deep, convnext_v2e_ablated_baseline,
    )
    variants = {
        "tiny": convnext_v2e_tiny,
        "wide": convnext_v2e_wide,
        "deep": convnext_v2e_deep,
        "ablated": convnext_v2e_ablated_baseline,
    }
    model = variants[args.variant]().cuda() if torch.cuda.is_available() else variants[args.variant]()

    print_summary(model)

    # FLOPs
    print(f"\n{'='*70}")
    print("FLOPs Estimation")
    print(f"{'='*70}")
    total_gflops = flop_count(model)
    heuristic = estimate_flops_formula(model)
    print(f"  [Heuristic] Estimated FLOPs: {heuristic:.2f} G")

    for mode in ["forward", "train"]:
        mult = 6 if mode == "train" else 2
        act = mult * count_params(model)
        print(f"  [Kaplan et al. {mode}] ~{mult}× params = {act/1e9:.2f} G")

    if not args.no_profile:
        profile_forward(model, batch_size=args.batch_size, steps=args.steps)
        profile_train_step(model, batch_size=args.batch_size, steps=args.steps)

    measure_throughput(model)
    profile_memory(model, batch_size=args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
