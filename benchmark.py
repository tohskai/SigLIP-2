import math
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from optimized_siglip2 import Siglip2SequenceVisionTransformerOptimized
from model import Siglip2SequenceVisionTransformer, Siglip2VisionConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and compare original vs optimized Siglip2 Vision Transformer"
    )
    parser.add_argument(
        "--batch", type=int, default=1024, help="Batch size (number of images)"
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=256,
        help="Maximum number of patches per image (must be a perfect square)",
    )
    parser.add_argument(
        "--mini-batch",
        type=int,
        default=64,
        help="Mini-Batch size (number of images being processed in 1 iteration)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Floating point precision",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run the benchmark on",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=20,
        help="Number of warmup iterations to stabilize performance",
    )
    parser.add_argument(
        "--benchmark-iterations",
        type=int,
        default=50,
        help="Number of iterations for the actual benchmark",
    )
    parser.add_argument(
        "--seed", type=int, default=1337, help="Random seed (default: 1337)"
    )
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
        help="Disable torch.compile (enabled by default)",
    )
    return parser.parse_args()


def benchmark_model(
    model,
    dataloader: DataLoader,
    warmup_iters: int,
    benchmark_iters: int,
    compile_enabled: bool,
    device: torch.device,
    modifier: int,
    label: str,
):
    # Optionally JIT compile
    if compile_enabled and device.type == "cuda":
        model = torch.compile(model)
    model.train()

    # Create an infinite iterator over the DataLoader
    data_iter = iter(dataloader)

    # Warmup forward + backward
    for _ in range(warmup_iters):
        try:
            batch_patches, batch_grids = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_patches, batch_grids = next(data_iter)

        # flatten patches into (batch*max_patches, embed_dim)
        bsz, max_patches, embed_dim = batch_patches.shape
        seq = batch_patches.view(bsz * max_patches, embed_dim)
        inputs = (seq, batch_grids)

        out = model(inputs)
        loss = out.sum()
        loss.backward()
        model.zero_grad()
        # clear gradients on inputs if any
        if batch_patches.requires_grad:
            batch_patches.grad = None

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark timing & memory
    fwd_times = []
    bwd_times = []
    torch.cuda.reset_peak_memory_stats(device)
    data_iter = iter(dataloader)
    for _ in range(benchmark_iters):
        # Get next batch
        try:
            batch_patches, batch_grids = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch_patches, batch_grids = next(data_iter)

        bsz, max_patches, embed_dim = batch_patches.shape
        seq = batch_patches.view(bsz * max_patches, embed_dim)
        inputs = (seq, batch_grids)

        torch.cuda.empty_cache()
        # Forward
        start_fwd = torch.cuda.Event(enable_timing=True)
        end_fwd = torch.cuda.Event(enable_timing=True)
        start_fwd.record()
        out = model(inputs)
        end_fwd.record()
        torch.cuda.synchronize()
        fwd_times.append(start_fwd.elapsed_time(end_fwd))

        torch.cuda.empty_cache()
        # Backward
        start_bwd = torch.cuda.Event(enable_timing=True)
        end_bwd = torch.cuda.Event(enable_timing=True)
        start_bwd.record()
        loss = out.sum()
        loss.backward()
        end_bwd.record()
        torch.cuda.synchronize()
        bwd_times.append(start_bwd.elapsed_time(end_bwd))

        model.zero_grad()
        if batch_patches.requires_grad:
            batch_patches.grad = None

    avg_fwd = (sum(fwd_times) / len(fwd_times)) * modifier
    avg_bwd = (sum(bwd_times) / len(bwd_times)) * modifier
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)  # MiB

    print(
        f"{label} -> Avg forward: {avg_fwd:.2f} ms | "
        f"Avg backward: {avg_bwd:.2f} ms | Peak Mem: {peak_mem:.1f} MiB"
    )
    return model


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device)

    # Initialize configuration and models
    config = Siglip2VisionConfig()
    orig_model = Siglip2SequenceVisionTransformer(config)
    opt_model = Siglip2SequenceVisionTransformerOptimized(config)
    orig_model.to(device=device, dtype=dtype)
    opt_model.to(device=device, dtype=dtype)

    # Compute dims
    embed_dim = config.num_channels * config.patch_size * config.patch_size
    grid_size = int(math.isqrt(args.max_patches))
    assert grid_size * grid_size == args.max_patches, (
        "max_patches must be a perfect square"
    )

    seq_patches = torch.randn(
        args.batch,
        args.max_patches,
        embed_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    token_grids = torch.full(
        (args.batch, 2), grid_size, device=device, dtype=torch.int64
    )

    dataset = TensorDataset(seq_patches, token_grids)
    dataloader = DataLoader(dataset, batch_size=args.mini_batch, shuffle=False)

    # Benchmark both models
    benchmark_model(
        orig_model,
        dataloader,
        args.warmup_iterations,
        args.benchmark_iterations,
        args.compile,
        device,
        args.batch // args.mini_batch,
        "Original",
    )
    benchmark_model(
        opt_model,
        dataloader,
        args.warmup_iterations,
        args.benchmark_iterations,
        args.compile,
        device,
        args.batch // args.mini_batch,
        "Optimized",
    )


if __name__ == "__main__":
    main()
