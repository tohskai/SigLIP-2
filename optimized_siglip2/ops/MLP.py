import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

from triton.language.extra.cuda.libdevice import tanh


@triton.jit
def gelu_approx(x):
    """
    GeLU_ activation - Gaussian error linear unit, with tanh approximation

    .. _GeLU: https://arxiv.org/pdf/1606.08415.pdf
    """
    return 0.5 * x * (1.0 + tanh(0.79788456760809 * x * (1.0 + 0.044715 * x * x)))


@triton.jit
def gelu_approx_grad(x):
    # CREDITS: Fast implementation proposed in
    # https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/fused_bias_gelu.py#L30
    tanh_out = tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    return 0.5 * x * (
        (1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)
    ) + 0.5 * (1 + tanh_out)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
        # good for int8
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "CACHE_KEY_K"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def kernel_linear_gelu_fwd(
    C,  # Pointers to matrices
    ACT_INPUT,
    A,
    B,
    bias,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_KEY_M,
    CACHE_KEY_N,
    CACHE_KEY_K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_cm,
    # stride_cn,  # Assume that stride_cn == 1
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    A_ROWMAJOR: tl.constexpr,
    B_COLMAJOR: tl.constexpr,
    SAVE_ACT_INPUT: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # trick to avoid masking on M and N axis
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    if A_ROWMAJOR:
        A = A + (ram[:, None] * stride_am + rk[None, :])
    else:
        A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    if B_COLMAJOR:
        B = B + (rk[:, None] + rbn[None, :] * stride_bn)
    else:
        B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        if A_ROWMAJOR:
            A += BLOCK_K
        else:
            A += BLOCK_K * stride_ak
        if B_COLMAJOR:
            B += BLOCK_K
        else:
            B += BLOCK_K * stride_bk

    bias = tl.load(bias + rn, mask=rn < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    if SAVE_ACT_INPUT:
        act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
        tl.store(act_in_ptrs, acc)

    acc = gelu_approx(acc)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc)


def triton_linear_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    save_act_input: bool = False,
) -> torch.Tensor:
    batch_shape, n = x.shape[:-1], x.shape[-1]
    batch_dim = batch_shape.numel()
    x_reshaped = x.reshape(batch_dim, n)

    if x_reshaped.stride(0) > 1 and x_reshaped.stride(1) > 1:
        x_reshaped = x_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()
    bias = bias.contiguous() if bias is not None else None

    assert x.dtype == weight.dtype, (
        f"Input and weight must have the same dtype, got {x.dtype} and {weight.dtype}"
    )
    if bias is not None:
        assert x.dtype == bias.dtype, (
            f"Input and bias must have the same dtype, got {x.dtype} and {bias.dtype}"
        )
    assert x_reshaped.shape[1] == weight.shape[1], (
        f"Incompatible dimensions: {x_reshaped.shape} - {weight.shape}"
    )

    assert bias is None or bias.shape[0] == weight.shape[0], (
        "Incompatible dimensions in between weight and bias"
    )

    M, K = x_reshaped.shape
    N, K = weight.shape

    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    act_input = torch.empty_like(output) if save_act_input else None

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )  # noqa

    kernel_linear_gelu_fwd[grid](
        output,
        act_input,
        x_reshaped,
        weight,  # data ptrs
        bias if bias is not None else x,  # auto skip bias if not present
        M,  # shapes
        N,
        K,
        M // 32,  # key for triton cache (limit number of compilations)
        N // 32,
        K // 32,
        stride_cm=output.stride(0),  # strides
        # stride_cn=output.stride(1),
        stride_am=x_reshaped.stride(0),
        stride_ak=x_reshaped.stride(1),
        stride_bk=weight.stride(1),
        stride_bn=weight.stride(0),
        SAVE_ACT_INPUT=save_act_input,
        A_ROWMAJOR=x_reshaped.stride(1) == 1,
        B_COLMAJOR=weight.stride(1) == 1,
        GROUP_M=8,  # speed optimization: group the programs
    )

    if not save_act_input:
        return output.reshape(*batch_shape, output.shape[-1])
    else:
        return (
            output.reshape(*batch_shape, output.shape[-1]),
            act_input.reshape(*batch_shape, act_input.shape[-1]),
        )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
        # good for int8
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "SPLIT_K": 1},
            num_stages=5,
            num_warps=2,
        ),
    ],
    key=["CACHE_KEY_M", "CACHE_KEY_N", "CACHE_KEY_K"],
)
@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % (args["BLOCK_K"] * args["SPLIT_K"]) == 0,
    }
)
@triton.jit
def kernel_linear_gelu_bwd(
    C,  # Pointers to matrices
    ACT_INPUT,
    A,
    B,
    # Matrix dimensions
    M,
    N,
    K,
    CACHE_KEY_M,
    CACHE_KEY_N,
    CACHE_KEY_K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. stride_am is how much to increase a_ptr
    # by to get the element one row down (A has M rows)
    stride_cm,
    # stride_cn,  # Assume that stride_cn == 1
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    # split k not used, not performant with activation, kept because early_config_prune is expecting it
    SPLIT_K: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    # now compute the block that each program will go through
    # rm (resp. rn) denotes a range of indices
    # for rows (resp. col) of C
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # trick to avoid masking on M and N axis
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(K, 0, -BLOCK_K):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            a = tl.load(A, mask=rk[None, :] < k, other=0.0)
            b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)

        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # optional: fused activation (while the data is in shared memory)
    act_in_ptrs = ACT_INPUT + ram[:, None] * stride_cm + rbn[None, :]
    act_input = tl.load(act_in_ptrs).to(acc.dtype)
    acc *= gelu_approx_grad(act_input)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # write back result
    C = C + rm[:, None] * stride_cm + rn[None, :]
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, acc, mask=mask)


def triton_linear_gelu_bwd(
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    act_input=None,
) -> torch.Tensor:
    """
    Compute e = activation(grad_output @ weight + bias).
    This wrapper kicks the `kernel_fwd` Triton kernel
    :param grad_output: input tensor
    :param weight: weight matrix
    :param activation: Activation name. Needs to be a Triton kernel.
    :param act_input: an optional tensor to save the activation inputs (for backward)
    :return: result tensor
    """
    batch_shape, n = grad_output.shape[:-1], grad_output.shape[-1]
    batch_dim = batch_shape.numel()
    grad_output_reshaped = grad_output.reshape(batch_dim, n)

    if grad_output_reshaped.stride(0) > 1 and grad_output_reshaped.stride(1) > 1:
        grad_output_reshaped = grad_output_reshaped.contiguous()
    if weight.stride(0) > 1 and weight.stride(1) > 1:
        weight = weight.contiguous()

    assert grad_output.dtype == weight.dtype, (
        f"grad_output and weight must have the same dtype, got {grad_output.dtype} and {weight.dtype}"
    )
    assert grad_output_reshaped.shape[1] == weight.shape[0], (
        f"Incompatible dimensions: {grad_output_reshaped.shape} - {weight.shape}"
    )

    # M, N, K in bwd are different from M, N, K in fwd
    M, K = grad_output_reshaped.shape
    K, N = weight.shape

    grad_input = torch.empty((M, N), device=grad_output.device, dtype=grad_output.dtype)

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )  # noqa

    kernel_linear_gelu_bwd[grid](
        grad_input,
        act_input,
        grad_output_reshaped,
        weight,  # data ptrs
        M,  # shapes
        N,
        K,
        M // 32,  # key for triton cache (limit number of compilations)
        N // 32,
        K // 32,
        stride_cm=grad_input.stride(0),  # strides
        # stride_cn=grad_input.stride(1),
        stride_am=grad_output_reshaped.stride(0),
        stride_ak=grad_output_reshaped.stride(1),
        stride_bk=weight.stride(0),
        stride_bn=weight.stride(1),
        GROUP_M=8,  # speed optimization: group the programs
    )

    return grad_input.reshape(*batch_shape, grad_input.shape[-1])


class FusedMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight1, bias1, weight2, bias2):
        x = x.contiguous()
        weight1 = weight1.contiguous()
        bias1 = bias1.contiguous()
        weight2 = weight2.contiguous()
        bias2 = bias2.contiguous()
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()
        output1 = triton_linear_gelu(x.reshape(batch_dim, n), weight1, bias1)
        output2 = F.linear(output1, weight2, bias2)

        ctx.save_for_backward(x, weight1, bias1, weight2)
        return output2.reshape(*batch_shape, output2.shape[-1])

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        x, weight1, bias1, weight2 = ctx.saved_tensors
        batch_shape, n = x.shape[:-1], x.shape[-1]
        batch_dim = batch_shape.numel()

        output1, act_input = triton_linear_gelu(
            x.reshape(batch_dim, n), weight1, bias1, save_act_input=True
        )

        grad_output = grad_output.reshape(batch_dim, grad_output.shape[-1])
        grad_weight2 = grad_output.t().matmul(output1)
        grad_bias2 = grad_output.sum(dim=0)

        grad_act_input = triton_linear_gelu_bwd(
            grad_output, weight2, act_input=act_input
        )

        grad_input = grad_act_input.matmul(weight1)
        grad_weight1 = grad_act_input.t().matmul(x.reshape(batch_dim, n))
        grad_bias1 = grad_act_input.sum(dim=0)

        return (
            grad_input.reshape_as(x),
            grad_weight1,
            grad_bias1,
            grad_weight2,
            grad_bias2,
            None,
        )


mlp_func = FusedMLPFunction.apply
