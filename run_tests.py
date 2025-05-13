import torch
import torch.nn as nn

from model import (
    Siglip2VisionConfig,
    Siglip2MLP,
    Siglip2SequenceEmbeddings,
    Siglip2Attention,
    Siglip2EncoderLayer,
)
from optimized_siglip2.model_optimized import (
    MLPImproved,
    LayerNormImproved,
    Siglip2SequenceEmbeddingsImproved,
    Siglip2AttentionImproved,
    Siglip2EncoderLayerImproved,
)


from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def print_metrics(metrics: dict):
    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Metric", style="bold")
    table.add_column("Original", justify="right")
    table.add_column("Improved", justify="right")
    for key, (orig, imp) in metrics.items():

        def fmt(x):
            if isinstance(x, torch.Tensor):
                x = x.item()
            return f"{x:,.2f}" if isinstance(x, float) else f"{x:,}"

        table.add_row(key, fmt(orig), fmt(imp))
    console.print(table)


def compare_models(
    test_name: str,
    org_model: torch.nn.Module,
    imp_model: torch.nn.Module,
    data: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device = torch.device("cuda"),
    warmup: int = 20,
    iters: int = 50,
    atol: float = 0.1,
    rtol: float = 0.0,
):
    org = org_model.to(device=device, dtype=dtype).train()
    imp = imp_model.to(device=device, dtype=dtype).train()

    imp.load_state_dict(org.state_dict(), strict=False)

    console.print(f"[bold]{test_name}:[/] ({dtype})")

    console.print("[bold]Correctness:[/]")
    out_org = org(data)
    out_imp = imp(data)

    def check(name: str, a: torch.Tensor, b: torch.Tensor):
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        mark = "[green]✔[/]" if ok else "[red]✖[/]"
        err = (a - b).abs().mean().item()
        console.print(f"{mark} {name} (avg error: {err:.2e})")

    check("Output", out_org, out_imp)

    org.zero_grad()
    imp.zero_grad()
    (out_org.sum()).backward()
    (out_imp.sum()).backward()
    for (n_org, p_org), (_, p_imp) in zip(
        org.named_parameters(), imp.named_parameters()
    ):
        if p_org.grad is not None:
            check(f"Grad {n_org}", p_org.grad, p_imp.grad)

    def run(model):
        m = model
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(warmup):
            (m(data).sum()).backward()
            m.zero_grad()
        fwd_times, bwd_times = [], []
        for _ in range(iters):
            torch.cuda.empty_cache()
            m.zero_grad()
            torch.cuda.reset_peak_memory_stats(device)
            fwd_start, fwd_end = torch.cuda.Event(True), torch.cuda.Event(True)
            bwd_start, bwd_end = torch.cuda.Event(True), torch.cuda.Event(True)
            fwd_start.record()
            out = m(data)
            fwd_end.record()
            torch.cuda.synchronize()
            fwd_times.append(fwd_start.elapsed_time(fwd_end))
            bwd_start.record()
            (out.sum()).backward()
            bwd_end.record()
            torch.cuda.synchronize()
            bwd_times.append(bwd_start.elapsed_time(bwd_end))
        return {
            "Fwd (ms)": sum(fwd_times) / len(fwd_times),
            "Bwd (ms)": sum(bwd_times) / len(bwd_times),
            "Peak Mem (MB)": torch.cuda.max_memory_allocated(device) / 1024**2,
        }

    metrics_org = run(org)
    metrics_imp = run(imp)
    combined = {k: (metrics_org[k], metrics_imp[k]) for k in metrics_org}

    print_metrics(combined)
    print(
        "============================================================================="
    )


def run(options):
    for name, org, imp, data, dtype in options:
        compare_models(name, org, imp, data, dtype)


def test_mlp(device, cfg, constant):
    options = [
        (
            "MLP",
            Siglip2MLP(cfg),
            MLPImproved(cfg),
            torch.randn(
                (constant, cfg.hidden_size), device=device, dtype=torch.bfloat16
            ),
            torch.bfloat16,
        ),
        (
            "MLP",
            Siglip2MLP(cfg),
            MLPImproved(cfg),
            torch.randn(
                (constant, cfg.hidden_size), device=device, dtype=torch.float16
            ),
            torch.float16,
        ),
    ]

    run(options)


def test_layernorm(device, cfg, constant):
    options = [
        (
            "LayerNorm",
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
            LayerNormImproved(cfg.hidden_size, eps=cfg.layer_norm_eps),
            torch.randn(
                (constant, cfg.hidden_size), device=device, dtype=torch.bfloat16
            ),
            torch.bfloat16,
        ),
        (
            "LayerNorm",
            nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps),
            LayerNormImproved(cfg.hidden_size, eps=cfg.layer_norm_eps),
            torch.randn(
                (constant, cfg.hidden_size), device=device, dtype=torch.float16
            ),
            torch.float16,
        ),
    ]

    run(options)


def test_embeddings(device, cfg, constant):
    batch = constant
    grid_size = int(cfg.num_patches**0.5)
    seq_sizes = torch.full((batch,), cfg.num_patches, dtype=torch.long, device=device)
    patch_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size
    total_patches = int(seq_sizes.sum().item())

    spatial_shapes = torch.tensor(
        [[grid_size, grid_size]] * batch,
        dtype=torch.long,
        device=device,
    )

    seq_patches_bf16 = torch.randn(
        total_patches, patch_dim, dtype=torch.bfloat16, device=device
    )
    seq_patches_f16 = torch.randn(
        total_patches, patch_dim, dtype=torch.float16, device=device
    )

    options = [
        (
            "Embeddings",
            Siglip2SequenceEmbeddings(cfg),
            Siglip2SequenceEmbeddingsImproved(cfg),
            (seq_patches_bf16, seq_sizes, spatial_shapes),
            torch.bfloat16,
        ),
        (
            "Embeddings",
            Siglip2SequenceEmbeddings(cfg),
            Siglip2SequenceEmbeddingsImproved(cfg),
            (seq_patches_f16, seq_sizes, spatial_shapes),
            torch.float16,
        ),
    ]

    run(options)


def test_encoder_layer(device, cfg, constant): ...


def main():
    constant = 20
    device = torch.device("cuda")
    cfg = Siglip2VisionConfig()
    cfg.attention_dropout = 0.0

    test_mlp(device, cfg, constant)
    test_layernorm(device, cfg, constant)
    test_embeddings(device, cfg, constant)

    # test_encoder_layer(device, cfg, constant)


if __name__ == "__main__":
    main()
