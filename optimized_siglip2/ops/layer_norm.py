import math

import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.language.extra.libdevice import rsqrt

def calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


@triton.jit
def _layer_norm_forward_kernel(
    Y_ptr,  # pointer to output, shape (n_rows, n_cols)
    Y_row_stride,  # stride of each row in output
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    X_row_stride,  # stride of each row in input
    W_ptr,  # pointer to weights, shape (n_cols,)
    W_row_stride,  # stride of each row in weights
    B_ptr,  # pointer to bias, shape (n_cols,)
    B_row_stride,  # stride of each row in bias
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    Mean_row_stride,  # stride of each row in mean
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    RSTD_row_stride,  # stride of each row in rstd
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y_ptr += row_idx * Y_row_stride
    X_ptr += row_idx * X_row_stride
    Mean_ptr += row_idx * Mean_row_stride
    RSTD_ptr += row_idx * RSTD_row_stride

    X_row = tl.load(X_ptr + col_offsets, mask=mask, other=0)
    W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
    B_row = tl.load(B_ptr + col_offsets, mask=mask, other=0)

    mean = tl.sum(X_row, axis=0) / n_cols
    Xmm = tl.where(mask, X_row - mean, 0)
    var = tl.sum(Xmm * Xmm, axis=0) / n_cols
    rstd = rsqrt(var + eps)

    tl.store(Mean_ptr, mean)
    tl.store(RSTD_ptr, rstd)

    Y_row = Xmm * rstd * W_row + B_row

    tl.store(Y_ptr + col_offsets, Y_row, mask=mask)


@triton.jit
def _layer_norm_backward_kernel(
    X_ptr,  # pointer to input, shape (n_rows, n_cols)
    W_ptr,  # pointer to weights, shape (n_cols,)
    Mean_ptr,  # pointer to mean, shape (n_rows,)
    RSTD_ptr,  # pointer to rstd, shape (n_rows,)
    DX_ptr,  # pointer to input grad, shape (n_rows, n_cols)
    DW_ptr,  # pointer to weights grad, shape (n_cols,)
    DB_ptr,  # pointer to bias grad, shape (n_cols,)
    DY_ptr,  # pointer to output grad, shape (n_rows, n_cols)
    stride_x,  # stride of each row in input
    stride_dx,  # stride of each row in input grad
    stride_dw,  # stride of each row in weights grad
    stride_db,  # stride of each row in bias grad
    stride_dy,  # stride of each row in output grad
    n_rows,
    n_cols,
    rows_per_program: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    dtype: tl.constexpr,
):
    """
    References:
    https://arxiv.org/abs/1607.06450
    https://github.com/karpathy/llm.c/blob/master/doc/layernorm/layernorm.md
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/layer_norm.py
    """
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = min((row_block_id + 1) * rows_per_program, n_rows)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    dw_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    db_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    X_ptr += row_start * stride_x
    Mean_ptr += row_start
    RSTD_ptr += row_start
    DX_ptr += row_start * stride_dx
    DY_ptr += row_start * stride_dy

    for _ in range(row_start, row_end):
        x = tl.load(X_ptr + cols, mask=mask, other=0.0)
        w = tl.load(W_ptr + cols, mask=mask, other=0.0)
        dy = tl.load(DY_ptr + cols, mask=mask, other=0.0)
        mean = tl.load(Mean_ptr)
        rstd = tl.load(RSTD_ptr)

        x_hat = (x - mean) * rstd
        wdy = w * dy
        c1 = tl.sum(x_hat * wdy, axis=0) / n_cols
        c2 = tl.sum(wdy, axis=0) / n_cols
        dx = (wdy - (x_hat * c1 + c2)) * rstd
        tl.store(DX_ptr + cols, dx.to(dtype), mask=mask)

        dw_row += dy * x_hat
        db_row += dy

        X_ptr += stride_x
        Mean_ptr += 1
        RSTD_ptr += 1
        DX_ptr += stride_dx
        DY_ptr += stride_dy

    tl.store(DW_ptr + row_block_id * stride_dw + cols, dw_row.to(dtype), mask=mask)
    tl.store(DB_ptr + row_block_id * stride_db + cols, db_row.to(dtype), mask=mask)


def layer_norm_forward(X, W, B, eps):
    shape = X.shape
    dim = shape[-1]
    X = X.view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    Mean = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    RSTD = torch.empty(n_rows, dtype=X.dtype, device=X.device)
    if X.shape[1] != W.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: input feature size (X.shape[1]={X.shape[1]}) "
            f"must match weight size (W.shape[0]={W.shape[0]})"
        )

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        kernel_args["grf_mode"] = "large"

    _layer_norm_forward_kernel[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        B,
        B.stride(0),
        Mean,
        Mean.stride(0),
        RSTD,
        RSTD.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        **kernel_args,  # XPU-specific optimization
    )
    return Y.view(*shape), X, Mean, RSTD, BLOCK_SIZE, num_warps


def layer_norm_backward(dY, X, W, B, Mean, RSTD):
    shape = dY.shape
    dim = shape[-1]
    dY = dY.view(-1, dim)
    n_rows, n_cols = dY.shape

    sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count

    DX = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    _DW = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)
    _DB = torch.empty((sm_count, n_cols), dtype=W.dtype, device=W.device)

    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    if n_cols > BLOCK_SIZE:
        raise RuntimeError(
            f"Feature dimension {n_cols} exceeds maximum supported size of {BLOCK_SIZE}. Consider using a smaller feature dimension."
        )

    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)
    triton_dtype = (
        tl.float32
        if X.dtype == torch.float32
        else tl.bfloat16
        if X.dtype == torch.bfloat16
        else tl.float16
        if X.dtype == torch.float16
        else tl.float32  # fallback to float32 for other types
    )

    # XPU-specific optimization
    kernel_args = {}
    if X.device.type == "xpu":
        kernel_args.update({"grf_mode": "large", "num_warps": 32, "num_stages": 4})

    _layer_norm_backward_kernel[grid](
        X,
        W,
        Mean,
        RSTD,
        DX,
        _DW,
        _DB,
        dY,
        X.stride(0),
        DX.stride(0),
        _DW.stride(0),
        _DB.stride(0),
        dY.stride(0),
        n_rows,
        n_cols,
        rows_per_program,
        BLOCK_SIZE=BLOCK_SIZE,
        dtype=triton_dtype,
        **kernel_args,  # XPU-specific optimization
    )

    DW = _DW.sum(dim=0).to(W.dtype)
    DB = _DB.sum(dim=0).to(W.dtype)

    DX = DX.view(*shape)
    return DX, DW, DB


class LigerLayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W, B, eps):
        Y, X, Mean, RSTD, BLOCK_SIZE, num_warps = layer_norm_forward(X, W, B, eps)
        ctx.save_for_backward(X, W, B, Mean, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        DX, DW, DB = layer_norm_backward(dY, X, W, B, Mean, RSTD)
        return DX, DW, DB, None


class LayerNormImproved(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, X):
        # X: (..., *normalized_shape)
        # Expand weight/bias to match X if necessary
        if self.elementwise_affine:
            W = self.weight
            B = self.bias
        else:
            # fall back to ones/zeros without tracking gradients
            W = torch.ones(self.normalized_shape, dtype=X.dtype, device=X.device)
            B = torch.zeros(self.normalized_shape, dtype=X.dtype, device=X.device)

        # torch.autograd.Function.apply takes the Function class itself as first arg
        return LigerLayerNormFunction.apply(X, W, B, self.eps)
