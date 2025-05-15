import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from random import randrange

@triton.autotune(
    configs=[
        triton.Config(kwargs={"BLOCK_SIZE": 64, "TILE_SIZE": 8}),
        triton.Config(kwargs={"BLOCK_SIZE": 128, "TILE_SIZE": 7}),
        triton.Config(kwargs={"BLOCK_SIZE": 256, "TILE_SIZE": 6}),
        triton.Config(kwargs={"BLOCK_SIZE": 512, "TILE_SIZE": 5}),
        triton.Config(kwargs={"BLOCK_SIZE": 1024, "TILE_SIZE": 4}),
        triton.Config(kwargs={"BLOCK_SIZE": 2048, "TILE_SIZE": 3}),
        triton.Config(kwargs={"BLOCK_SIZE": 4096, "TILE_SIZE": 2}),
    ],
    key=["n_elements"],
)
@triton.jit
def _seeded_dropout(
    x_ptr,
    output_ptr,
    n_elements,
    p,
    seed,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0) * TILE_SIZE
    for _ in tl.static_range(0, TILE_SIZE):
        pid += 1
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        # load data from x
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        # randomly prune it
        random = tl.rand(seed, offsets)
        x_keep = random > p
        # write-back
        output = tl.where(x_keep, x / (1 - p), 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    x = x.contiguous()
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements = x.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"] * meta["TILE_SIZE"]),
    )
    _seeded_dropout[grid](x, output, n_elements, p, seed)
    return output


class ImprovedDropoutFunction(torch.autograd.Function):
    @classmethod
    def forward(cls, ctx, x, p):
        seed = randrange(int(1e6))
        ctx.p = p
        ctx.seed = seed
        return seeded_dropout(x, p, seed)

    @classmethod
    def backward(cls, ctx, dy):
        p = ctx.p
        seed = ctx.seed
        return seeded_dropout(dy, p, seed), None


dropout_func = ImprovedDropoutFunction.apply
