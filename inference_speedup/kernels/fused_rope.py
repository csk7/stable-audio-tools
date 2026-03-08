"""
Triton fused RoPE (Rotary Position Embedding) kernel.

Fuses the full apply_rotary_pos_emb call in one pass:
  fp32 cast → cos*t1 - sin*t2  (first half)
            → sin*t1 + cos*t2  (second half)  [rotate_half semantics]
  → concat unrotated → cast back to input dtype

Input shape:  (B, H, N, D) — q or k tensor
freqs shape:  (N, rot_dim) — fp32, rot_dim = D_head//2 for this model

Partial rotation: rot_dim <= D. For this DiT: D=64, rot_dim=32 (half=16).
One Triton program per (b, h, n) row.  Grid = B*H*N = 2*24*1025 = 49,200 programs.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_kernel(T, Freqs, Out, stride_th, stride_tn, stride_fn, N, D: tl.constexpr,
    rot_dim: tl.constexpr, HALF: tl.constexpr, UNROT: tl.constexpr, STORE_FP16: tl.constexpr):
    """
    One program per row (b, h, n).  Loads D elements, applies RoPE to first rot_dim,
    copies remaining UNROT elements unchanged, stores back in original dtype.
    """
    pid = tl.program_id(0)
    # pid layout: pid = bh * N + n, where bh = b*H + h
    n_   = pid % N
    bh_  = pid // N

    T_row   = T   + bh_ * stride_th + n_ * stride_tn
    Out_row = Out + bh_ * stride_th + n_ * stride_tn
    F_row   = Freqs + n_ * stride_fn

    # Column indices
    c0 = tl.arange(0, HALF)           # [0..HALF=16)
    c1 = HALF + tl.arange(0, HALF)    # [HALF..rot_dim)

    # Load t_rot first/second halves → fp32
    t1 = tl.load(T_row + c0).to(tl.float32)   
    t2 = tl.load(T_row + c1).to(tl.float32)   

    # Load only the first half and reuse.
    f = tl.load(F_row + c0)

    cos_f = tl.cos(f)
    sin_f = tl.sin(f)

    # Apply rotate_half: rotate_half(t_rot) = cat(-t2, t1)
    out1 = t1 * cos_f - t2 * sin_f
    out2 = t2 * cos_f + t1 * sin_f

    if STORE_FP16:
        tl.store(Out_row + c0, out1.to(tl.float16))
        tl.store(Out_row + c1, out2.to(tl.float16))
    else:
        tl.store(Out_row + c0, out1)
        tl.store(Out_row + c1, out2)

    # Copy unrotated part unchanged
    if UNROT > 0:
        c_unrot = rot_dim + tl.arange(0, UNROT)
        x_unrot = tl.load(T_row + c_unrot).to(tl.float32)
        if STORE_FP16:
            tl.store(Out_row + c_unrot, x_unrot.to(tl.float16))
        else:
            tl.store(Out_row + c_unrot, x_unrot)


def apply_rope_fused(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Fused RoPE matching apply_rotary_pos_emb semantics (partial rotation, scale=1).

    t:     (B, H, N, D) — fp16 or bf16, will be made contiguous if needed
    freqs: (N, rot_dim) — fp32 rot_dim = D//2
    """
    assert t.ndim == 4, f"Expected (B, H, N, D), got {t.shape}"
    assert freqs.ndim == 2, f"Expected (N, rot_dim) for freqs, got {freqs.shape}"
    if not t.is_contiguous():
        t = t.contiguous()
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()

    B, H, N, D = t.shape
    rot_dim = freqs.shape[-1]
    half = rot_dim // 2
    unrot = D - rot_dim

    assert rot_dim % 2 == 0, f"rot_dim={rot_dim} must be even"

    out = torch.empty_like(t)
    store_fp16 = t.dtype == torch.float16

    # For (B,H,N,D) contiguous: strides = (H*N*D, N*D, D, 1)
    _, stride_th, stride_tn, _ = t.stride()
    _, stride_fn = freqs.stride()

    grid = (B * H * N,)

    _rope_fwd_kernel[grid](t, freqs, out, stride_th=stride_th, stride_tn=stride_tn, stride_fn=stride_fn,
        N=N, D=D, rot_dim=rot_dim, HALF=half, UNROT=unrot, STORE_FP16=store_fp16, num_warps=2, num_stages=2)
    return out
