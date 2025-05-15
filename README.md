# SigLIP-2

## Running the tests

```bash
python run_tests.py
```

## Running the profiler

```bash
python profile_siglip2.py \
  --batch 64 \
  --max-patches 1024 \
  --dtype bf16 \
  --device cuda \
  --warmup-steps 10 \
  --profile-steps 20 \
  --trace-file bf16_trace.json.gz

```
## Command-Line Arguments

This script benchmarks and compares the original and optimized versions of the Siglip2 Vision Transformer.

### Available Arguments

| Argument                  | Type   | Default | Description                                                                 |
|---------------------------|--------|---------|-----------------------------------------------------------------------------|
| `--batch`                 | int    | 1024    | Total number of images processed in the full benchmark pass.               |
| `--max-patches`           | int    | 256     | Maximum number of patches per image (must be a perfect square).            |
| `--mini-batch`            | int    | 64      | Number of images processed per iteration. Useful for memory management.    |
| `--dtype`                 | str    | `fp32`  | Precision type. Options: `fp32`, `fp16`, `bf16`.                           |
| `--device`                | str    | `cuda`  | Device to run on. Options: `cuda`, `cpu`.                                  |
| `--warmup-iterations`     | int    | 20      | Number of warmup iterations before benchmarking begins.                    |
| `--benchmark-iterations`  | int    | 50      | Number of iterations used for actual benchmarking.                         |
| `--seed`                  | int    | 1337    | Random seed for reproducibility.                                           |
| `--no-compile`            | flag   |         | Disable `torch.compile`. Enabled by default unless this flag is set.       |

### Example Usage

```bash
python benchmark.py --batch 1024 --max-patches 256 --dtype bf16 --device cuda
```
