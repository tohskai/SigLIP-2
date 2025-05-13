import argparse
import math
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from optimized_siglip2 import (
    Siglip2SequenceVisionTransformerOptimized,
    Siglip2VisionConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile Siglip2SequenceVisionTransformerOptimized (forward + backward)"
    )
    parser.add_argument(
        "--batch", type=int, default=32, help="Number of images in the batch"
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=256,
        help="Number of patches per image (must be a perfect square)",
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
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=5, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--profile-steps", type=int, default=10, help="Number of profiling iterations"
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default="trace.json.gz",
        help="Output Chrome trace file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    # Instantiate model
    config = Siglip2VisionConfig()
    model = Siglip2SequenceVisionTransformerOptimized(config)
    model = torch.compile(model)
    model.to(device=device, dtype=dtype)
    model.train()  # enable gradient tracking

    # Synthetic input creation
    patch_dim = config.num_channels * config.patch_size * config.patch_size
    grid_size = int(math.sqrt(args.max_patches))
    if grid_size * grid_size != args.max_patches:
        raise ValueError("--max-patches must be a perfect square")

    batch = args.batch
    token_grids = torch.tensor(
        [[grid_size, grid_size] for _ in range(batch)], dtype=torch.int64, device=device
    )
    seq_len = batch * args.max_patches
    seq_patches = torch.randn(
        seq_len,
        patch_dim,
        dtype=dtype,
        device=device,
        requires_grad=False,  # inputs as non-trainable
    )

    # Warmup runs (forward only)
    for _ in range(args.warmup_steps):
        _ = model((seq_patches, token_grids))

    # Profiling forward + backward
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
        use_cuda=(args.device == "cuda"),
    ) as prof:
        for _ in range(args.profile_steps):
            model.zero_grad()
            # Forward pass
            with record_function("model_forward"):
                outputs = model((seq_patches, token_grids))
            # Simple scalar loss
            loss = outputs.float().mean()
            # Backward pass
            with record_function("model_backward"):
                loss.backward()

    # Print profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Export chrome trace
    prof.export_chrome_trace(args.trace_file)
    print(f"Chrome trace saved to {args.trace_file}")


if __name__ == "__main__":
    main()
