"""
Triton fused residual-add + LayerNorm.
Fuses both: out_norm = layernorm(residual + sublayer_out), out_sum = residual + sublayer_out.
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def _residual_add_layernorm_fwd_kernel_single_row(
    Residual,
    SublayerOut,
    OutNorm,
    OutSum,
    Weight,
    Bias,
    stride,
    N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STORE_FP16: tl.constexpr,
):
    """Single-row: one program per row (no Weight/Bias reuse)."""
    row = tl.program_id(0)
    Residual += row * stride
    SublayerOut += row * stride
    OutNorm += row * stride
    OutSum += row * stride

    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        r = tl.load(Residual + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(SublayerOut + cols, mask=mask, other=0.0).to(tl.float32)
        x = tl.where(mask, r + s, 0.0)
        _sum += x
        _sum_sq += x * x
    mean = tl.sum(_sum, axis=0) / N
    mean_sq = tl.sum(_sum_sq, axis=0) / N
    var = mean_sq - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        r = tl.load(Residual + cols, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(SublayerOut + cols, mask=mask, other=0.0).to(tl.float32)
        r_plus_s = r + s
        w = tl.load(Weight + cols, mask=mask)
        b = tl.load(Bias + cols, mask=mask)
        x_hat = (r_plus_s - mean) * rstd
        y = x_hat * w + b
        if STORE_FP16:
            tl.store(OutNorm + cols, y.to(tl.float16), mask=mask)
        else:
            tl.store(OutNorm + cols, y, mask=mask)
        tl.store(OutSum + cols, r_plus_s, mask=mask)


@triton.jit
def _residual_add_layernorm_fwd_kernel(
    Residual,
    SublayerOut,
    OutNorm,
    OutSum,
    Weight,
    Bias,
    stride,
    M,
    N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    STORE_FP16: tl.constexpr,
):
    """Fused: OutNorm = layernorm(Residual + SublayerOut), OutSum = Residual + SublayerOut.
    Two-row interleaved: both rows' data loaded in the same loop body for ILP.
    Weight/Bias loaded once per block and shared by both rows.
    """
    row0 = tl.program_id(0) * 2
    row1 = row0 + 1
    has_row1 = row1 < M

    R0 = Residual + row0 * stride
    S0 = SublayerOut + row0 * stride
    R1 = Residual + row1 * stride
    S1 = SublayerOut + row1 * stride

    # Interleaved stats pass: load both rows in the same loop iteration
    _sum0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_sq0 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    _sum_sq1 = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        r0 = tl.load(R0 + cols, mask=mask, other=0.0).to(tl.float32)
        s0 = tl.load(S0 + cols, mask=mask, other=0.0).to(tl.float32)
        x0 = tl.where(mask, r0 + s0, 0.0)
        _sum0 += x0
        _sum_sq0 += x0 * x0
        if has_row1:
            r1 = tl.load(R1 + cols, mask=mask, other=0.0).to(tl.float32)
            s1 = tl.load(S1 + cols, mask=mask, other=0.0).to(tl.float32)
            x1 = tl.where(mask, r1 + s1, 0.0)
            _sum1 += x1
            _sum_sq1 += x1 * x1

    mean0 = tl.sum(_sum0, axis=0) / N
    mean_sq0 = tl.sum(_sum_sq0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(mean_sq0 - mean0 * mean0 + eps)

    if has_row1:
        mean1 = tl.sum(_sum1, axis=0) / N
        mean_sq1 = tl.sum(_sum_sq1, axis=0) / N
        rstd1 = 1.0 / tl.sqrt(mean_sq1 - mean1 * mean1 + eps)
    else:
        mean1 = 0.0
        rstd1 = 1.0

    ON0 = OutNorm + row0 * stride
    OS0 = OutSum + row0 * stride
    ON1 = OutNorm + row1 * stride
    OS1 = OutSum + row1 * stride

    # Interleaved output pass: W/B loaded once, applied to both rows
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(Weight + cols, mask=mask)
        b = tl.load(Bias + cols, mask=mask)

        r0 = tl.load(R0 + cols, mask=mask, other=0.0).to(tl.float32)
        s0 = tl.load(S0 + cols, mask=mask, other=0.0).to(tl.float32)
        rps0 = r0 + s0
        y0 = (rps0 - mean0) * rstd0 * w + b
        if STORE_FP16:
            tl.store(ON0 + cols, y0.to(tl.float16), mask=mask)
        else:
            tl.store(ON0 + cols, y0, mask=mask)
        tl.store(OS0 + cols, rps0, mask=mask)

        if has_row1:
            r1 = tl.load(R1 + cols, mask=mask, other=0.0).to(tl.float32)
            s1 = tl.load(S1 + cols, mask=mask, other=0.0).to(tl.float32)
            rps1 = r1 + s1
            y1 = (rps1 - mean1) * rstd1 * w + b
            if STORE_FP16:
                tl.store(ON1 + cols, y1.to(tl.float16), mask=mask)
            else:
                tl.store(ON1 + cols, y1, mask=mask)
            tl.store(OS1 + cols, rps1, mask=mask)


def residual_add_layernorm(
    residual: torch.Tensor,
    sublayer_out: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused residual add + LayerNorm. Returns (out_norm, out_sum).
    Set USE_PYTORCH_FUSED=1 to fallback to PyTorch (add + F.layer_norm) if issues.
    """
    assert residual.shape == sublayer_out.shape
    assert residual.is_contiguous() and sublayer_out.is_contiguous()

    if os.environ.get("USE_PYTORCH_FUSED", "0") == "1":
        out_sum = (residual + sublayer_out).contiguous()
        out_norm = torch.nn.functional.layer_norm(
            out_sum, out_sum.shape[-1:], weight=weight, bias=bias, eps=eps
        )
        return out_norm, out_sum

    # Allocate outputs - OutNorm in input dtype when fp16 (saves bandwidth), OutSum matches input
    store_fp16 = residual.dtype == torch.float16
    out_norm = torch.empty(residual.shape, dtype=residual.dtype, device=residual.device)
    out_sum = torch.empty_like(residual)

    x = residual.reshape(-1, residual.shape[-1])
    M, N = x.shape
    stride = x.stride(0)

    # Allow BLOCK_SIZE up to 2048 for N=1536 (1 iteration per pass vs 2)
    BLOCK_SIZE = min(2048, triton.next_power_of_2(N))
    num_warps = 8
    use_multi_row = os.environ.get("RESIDUAL_LN_MULTI_ROW", "0") == "1"
    num_stages = int(os.environ.get("RESIDUAL_LN_NUM_STAGES", "2"))

    if use_multi_row:
        num_programs = (M + 1) // 2
        _residual_add_layernorm_fwd_kernel[(num_programs,)](
            residual,
            sublayer_out,
            out_norm,
            out_sum,
            weight,
            bias,
            stride=stride,
            M=M,
            N=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
            STORE_FP16=store_fp16,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _residual_add_layernorm_fwd_kernel_single_row[(M,)](
            residual,
            sublayer_out,
            out_norm,
            out_sum,
            weight,
            bias,
            stride=stride,
            N=N,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE,
            STORE_FP16=store_fp16,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    return out_norm, out_sum
