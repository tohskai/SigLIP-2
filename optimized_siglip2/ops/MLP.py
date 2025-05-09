import triton
import triton.language as tl

from triton.language.extra.cuda.libdevice import tanh

import torch
import torch.nn as nn


@triton.jit
def gelu(input):
    input_cube = input * input * input
    inner = 0.7978845608 * (input + 0.044715 * input_cube)
    cdf = 0.5 * (1 + tanh(inner))
    return cdf * input


@triton.jit
def gelu_grad(input):
    """
    Calculates the gradient of GELU.

    Args:
        input: Input. The input must be loaded and cannot be a pointer.

    Returns:
        Gradient of GELU.
    """
    input = input.to(tl.float32)
    cdf = 0.5 * (1 + tl.math.erf(0.707106781 * input))
    cdf_grad = 0.39894228 * tl.exp(-0.5 * input * input)
    return cdf_grad * input + cdf


# triton.heuristics can be used either standalone or
# before triton.autotune, it cannot be used
# after triton.autotune. This implies that
# if triton.heuristics and triton.autotune are to be used together,
# triton.heuristics must be used first.
@triton.heuristics({"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0})
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
    ],
    key=["size"],
)
@triton.jit
def gelu_func_backward_kernel(
    output_grad_pointer,
    input_pointer,
    input_grad_pointer,
    size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < size

    output_grad = tl.load(output_grad_pointer + offset, mask=mask)
    out = gelu_grad(tl.load(input_pointer + offset, mask=mask))

    tl.store(input_grad_pointer + offset, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 64,
            },
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 256,
                "BLOCK_SIZE_K": 32,
            },
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 32,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 128,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 32,
                "BLOCK_SIZE_K": 32,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 32,
            },
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def triton_matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    c_after_gelu_ptr,
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    stride_bias: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,  #
    USE_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    if USE_BIAS:
        bias_ptrs = bias_ptr + offs_am * stride_bias
        bias = tl.load(
            bias_ptrs, mask=offs_am < M, other=0, eviction_policy="evict_last"
        )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0)

        accumulator += tl.dot(a, b, allow_tf32=True)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if USE_BIAS:
        accumulator += bias[:, None]

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_after_gelu_ptrs = (
        c_after_gelu_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
    accumulator = gelu(accumulator)
    tl.store(c_after_gelu_ptrs, accumulator, mask=c_mask)


def matmul(a, b, bias=None):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    c_after_gelu = torch.empty((M, N), device=a.device, dtype=a.dtype)

    use_bias = bias is not None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    triton_matmul_kernel[grid](
        a,
        b,
        c,
        bias,
        c_after_gelu,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        bias.stride(0) if bias is not None else None,
        USE_BIAS=use_bias,
    )
    return c, c_after_gelu


class LinearGelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, b):
        # z = x @ W^T + b
        z, gelu_z = matmul(x, W.transpose(-2, -1), bias=b)
        ctx.save_for_backward(x, W, z)
        ctx.output_type = gelu_z.dtype
        return gelu_z

    @staticmethod
    def backward(ctx, grad_output):
        x, W, z = ctx.saved_tensors

        # ---- GELU derivative (approximate) ----
        sqrt2_over_pi = 0.7978845608
        u = sqrt2_over_pi * (z + 0.044715 * z**3)
        tanh_u = u.tanh()
        gelu_grad = (
            0.5 * (1 + tanh_u)
            + 0.5 * z * (1 - tanh_u**2) * sqrt2_over_pi * (1 + 3 * 0.044715 * z**2)
        ).to(ctx.output_type)

        # ---- chain it ----
        grad_z = grad_output * gelu_grad

        # ---- linear layer grads ----
        grad_x = grad_z.matmul(W) if ctx.needs_input_grad[0] else None
        grad_W = grad_z.t().matmul(x) if ctx.needs_input_grad[1] else None
        grad_b = grad_z.sum(dim=0) if ctx.needs_input_grad[2] else None

        return grad_x, grad_W, grad_b


class MLPImproved(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(LinearGelu.apply(hidden_states, self.fc1.weight, self.fc1.bias))
