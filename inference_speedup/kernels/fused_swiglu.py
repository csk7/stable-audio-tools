"""
Triton fused SwiGLU activation kernel.
Fuses chunk + SiLU(gate) * x after a Linear proj that outputs (B, N, 2*D).
Input shape: (B, N, 2*D) — post-GEMM from GLU.proj
Output shape: (B, N, D) — SiLU(gate) * x

For DiT shapes: input (2, 1025, 12288) -> output (2, 1025, 6144).
Called ~2400 times (24 layers × 100 steps). Memory-bandwidth bound.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _swiglu_fwd_kernel(Input, Output, stride_in, stride_out, D, BLOCK_D: tl.constexpr, STORE_FP16: tl.constexpr):
    """
    One program per row. Input row has 2*D elements: [x_half | gate_half].
    Output row has D elements: SiLU(gate_half) * x_half.
    Iterates over D in BLOCK_D-sized chunks to handle any D.
    """
    row = tl.program_id(0)
    In_row = Input + row * stride_in
    Out_row = Output + row * stride_out

    for d_start in range(0, D, BLOCK_D):
        cols = d_start + tl.arange(0, BLOCK_D)
        mask = cols < D

        x    = tl.load(In_row + cols,     mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(In_row + D + cols, mask=mask, other=0.0).to(tl.float32)

        silu_gate = gate * tl.sigmoid(gate)
        out = x * silu_gate

        if STORE_FP16:
            tl.store(Out_row + cols, out.to(tl.float16), mask=mask)
        else:
            tl.store(Out_row + cols, out, mask=mask)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU: splits last dim in half, applies SiLU to gate, returns gate * x.
    x: (..., 2*D) contiguous tensor (post-proj output)
    Returns: (..., D) tensor
    """
    assert x.shape[-1] % 2 == 0, "Last dim must be even for SwiGLU split"
    orig_shape = x.shape
    D = x.shape[-1] // 2

    x_flat = x.reshape(-1, x.shape[-1])
    if not x_flat.is_contiguous():
        x_flat = x_flat.contiguous()
    M = x_flat.shape[0]

    out_flat = torch.empty((M, D), dtype=x.dtype, device=x.device)

    # BLOCK_D=4096 gives 2 loop iterations for D=6144, 1 for D<=4096.
    BLOCK_D = 4096
    store_fp16 = x.dtype == torch.float16

    _swiglu_fwd_kernel[(M,)]( x_flat, out_flat, stride_in=x_flat.stride(0),
        stride_out=out_flat.stride(0), D=D, BLOCK_D=BLOCK_D, STORE_FP16=store_fp16, num_warps=8, num_stages=2)

    out_shape = orig_shape[:-1] + (D,)
    return out_flat.reshape(out_shape)
