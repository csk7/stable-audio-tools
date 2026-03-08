"""
sa_open_v6: Cross-attention KV cache + fused residual+layernorm + fused SwiGLU + fused RoPE.
- Cross-attention KV caching from v3
- Triton fused residual_add_layernorm from v4
- Triton fused SwiGLU activation from v5
- Triton fused RoPE: replaces apply_rotary_pos_emb (fp32 cast+rotate_half+cat fused)
"""
import gc
import os
import sys
import time

_inference_speedup_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _inference_speedup_dir not in sys.path:
    sys.path.insert(0, _inference_speedup_dir)

import torch
import soundfile as sf
from torch.profiler import profile, record_function, ProfilerActivity

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.sampling import sample_k
from stable_audio_tools.models import transformer as sa_transformer
from goldens_save import verify_golden_match
from tools.tools import patch_attention_with_sdpa_flash, setup_torch_backend, build_conditioning
from tools.tools import build_dit_kwargs, create_noise, normalize_audio, print_timing

from kernels.fused_residual_layernorm import residual_add_layernorm
from kernels.fused_swiglu import swiglu as fused_swiglu
from kernels.fused_rope import apply_rope_fused

_orig_attention_forward = sa_transformer.Attention.forward


# ---------------------------------------------------------------------------
# Fused RoPE: replace apply_rotary_pos_emb in the transformer module namespace.
# This affects all call-sites inside transformer.py (self-attn and cross-attn).
# ---------------------------------------------------------------------------

_orig_apply_rotary_pos_emb = sa_transformer.apply_rotary_pos_emb

def _apply_rotary_pos_emb_fused(t, freqs, scale=1):
    """
    Drop-in replacement for apply_rotary_pos_emb.
    Falls back to the original implementation when:
      - scale != 1 (xpos path)
      - t is not 4-D (unexpected shape)
      - freqs is not 2-D (batched freqs path)
      - shapes don't satisfy the power-of-2 constraints required by the kernel
    """
    if scale != 1:
        return _orig_apply_rotary_pos_emb(t, freqs, scale=scale)
    if t.ndim != 4 or freqs.ndim != 2:
        return _orig_apply_rotary_pos_emb(t, freqs, scale=scale)

    B, H, N, D = t.shape

    freqs = freqs[-N:]
    if not freqs.is_contiguous():
        freqs = freqs.contiguous()

    out_dtype = t.dtype
    if out_dtype not in (torch.float16, torch.bfloat16):
        t = t.to(torch.float16)

    if not t.is_contiguous():
        t = t.contiguous()

    out = apply_rope_fused(t, freqs)
    return out if out_dtype in (torch.float16, torch.bfloat16) else out.to(out_dtype)

sa_transformer.apply_rotary_pos_emb = _apply_rotary_pos_emb_fused


# ---------------------------------------------------------------------------
# Cross-attention KV cache (unchanged from v4/v5)
# ---------------------------------------------------------------------------

def _attention_forward_with_cross_kv_cache(self, x, context=None, rotary_pos_emb=None, causal=None,
    flex_attention_block_mask=None, flex_attention_score_mod=None, flash_attn_sliding_window=None):
    if context is None or (not hasattr(self, "to_q")):
        return _orig_attention_forward(self, x, context=context, rotary_pos_emb=rotary_pos_emb, causal=causal,
            flex_attention_block_mask=flex_attention_block_mask, flex_attention_score_mod=flex_attention_score_mod,
            flash_attn_sliding_window=flash_attn_sliding_window)

    h, kv_h = self.num_heads, self.kv_heads
    q = self.to_q(x)
    q = sa_transformer.rearrange(q, "b n (h d) -> b h n d", h=h)

    cached = getattr(self, "_cross_kv_cached", None)
    if cached is not None:
        k, v = cached
    else:
        k, v = self.to_kv(context).chunk(2, dim=-1)
        k = sa_transformer.rearrange(k, "b n (h d) -> b h n d", h=kv_h)
        v = sa_transformer.rearrange(v, "b n (h d) -> b h n d", h=kv_h)

        if self.num_heads != self.kv_heads:
            heads_per_kv_head = self.num_heads // self.kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

        self._cross_kv_cached = (k, v)

    if self.qk_norm == "l2":
        q = sa_transformer.F.normalize(q, dim=-1)
        k = sa_transformer.F.normalize(k, dim=-1)
    elif self.qk_norm != "none":
        q, k = self.apply_qk_layernorm(q, k)

    if rotary_pos_emb is not None:
        freqs, _ = rotary_pos_emb
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        freqs = freqs.to(torch.float32)
        if q.shape[-2] >= k.shape[-2]:
            ratio = q.shape[-2] / k.shape[-2]
            q_freqs, k_freqs = freqs, ratio * freqs
        else:
            ratio = k.shape[-2] / q.shape[-2]
            q_freqs, k_freqs = ratio * freqs, freqs
        q = sa_transformer.apply_rotary_pos_emb(q, q_freqs).to(v.dtype)
        k = sa_transformer.apply_rotary_pos_emb(k, k_freqs).to(v.dtype)

    n = q.shape[-2]
    causal = self.causal if causal is None else causal
    if n == 1 and causal:
        causal = False

    out = self.apply_attn(q, k, v, causal=causal, flex_attention_block_mask=flex_attention_block_mask,
        flex_attention_score_mod=flex_attention_score_mod, flash_attn_sliding_window=flash_attn_sliding_window)

    out = sa_transformer.rearrange(out, "b h n d -> b n (h d)")
    out = self.to_out(out)

    if self.feat_scale:
        out_dc = out.mean(dim=-2, keepdim=True)
        out_hf = out - out_dc
        out = out + self.lambda_dc * out_dc + self.lambda_hf * out_hf

    return out

def clear_cross_attention_kv_cache(denoiser):
    for module in denoiser.modules():
        if isinstance(module, sa_transformer.Attention) and hasattr(module, "_cross_kv_cached"):
            module._cross_kv_cached = None

sa_transformer.Attention.forward = _attention_forward_with_cross_kv_cache
patch_attention_with_sdpa_flash(sa_transformer)


# ---------------------------------------------------------------------------
# Fused residual+LayerNorm (unchanged from v4/v5)
# ---------------------------------------------------------------------------

def _can_use_fused_residual_ln(norm):
    return (isinstance(norm, sa_transformer.LayerNorm) and not getattr(norm, "force_fp32", False))

_orig_transformer_block_forward = sa_transformer.TransformerBlock.forward

def _transformer_block_forward_fused_residual_ln(self, x, context=None, global_cond=None, rotary_pos_emb=None,
    self_attention_block_mask=None, self_attention_score_mod=None, cross_attention_block_mask=None,
    cross_attention_score_mod=None, self_attention_flash_sliding_window=None,
    cross_attention_flash_sliding_window=None):
    if rotary_pos_emb is None and self.add_rope:
        rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

    if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
        return _orig_transformer_block_forward(
            self, x, context=context, global_cond=global_cond, rotary_pos_emb=rotary_pos_emb,
            self_attention_block_mask=self_attention_block_mask, self_attention_score_mod=self_attention_score_mod,
            cross_attention_block_mask=cross_attention_block_mask, cross_attention_score_mod=cross_attention_score_mod,
            self_attention_flash_sliding_window=self_attention_flash_sliding_window,
            cross_attention_flash_sliding_window=cross_attention_flash_sliding_window)

    use_fused_self = _can_use_fused_residual_ln(self.cross_attend_norm if self.cross_attend else self.ff_norm)
    use_fused_cross = self.cross_attend and _can_use_fused_residual_ln(self.ff_norm)

    attn_out = self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb=rotary_pos_emb,
        flex_attention_block_mask=self_attention_block_mask, flex_attention_score_mod=self_attention_score_mod,
        flash_attn_sliding_window=self_attention_flash_sliding_window))
    if use_fused_self:
        next_norm = self.cross_attend_norm if self.cross_attend else self.ff_norm
        x_norm, x_residual = residual_add_layernorm(x, attn_out, next_norm.gamma, next_norm.beta, eps=next_norm.eps)
    else:
        x = x + attn_out
        x_norm, x_residual = None, None

    if context is not None and self.cross_attend:
        cross_attn_input = x_norm if use_fused_self else self.cross_attend_norm(x)
        cross_out = self.cross_attn_scale(self.cross_attn(cross_attn_input, context=context,
            flex_attention_block_mask=cross_attention_block_mask, flex_attention_score_mod=cross_attention_score_mod,
            flash_attn_sliding_window=cross_attention_flash_sliding_window))
        r = x_residual if use_fused_self else x
        if use_fused_cross:
            x_norm, x_residual = residual_add_layernorm(r, cross_out, self.ff_norm.gamma, self.ff_norm.beta, eps=self.ff_norm.eps)
        else:
            x = r + cross_out
            x_norm, x_residual = None, None

    if self.conformer is not None:
        r = x_residual if (use_fused_self or use_fused_cross) else x
        x = r + self.conformer_scale(self.conformer(r))
        x_residual, x_norm = x, None

    r = x_residual if (use_fused_self or use_fused_cross) else x
    ff_input = x_norm if use_fused_cross else self.ff_norm(r)
    x = r + self.ff_scale(self.ff(ff_input))
    return x

sa_transformer.TransformerBlock.forward = _transformer_block_forward_fused_residual_ln


# ---------------------------------------------------------------------------
# Fused SwiGLU (unchanged from v5)
# ---------------------------------------------------------------------------

_orig_glu_forward = sa_transformer.GLU.forward

def _glu_forward_fused_swiglu(self, x):
    if self.use_conv:
        return _orig_glu_forward(self, x)
    if not isinstance(self.act, torch.nn.SiLU):
        return _orig_glu_forward(self, x)
    proj_out = self.proj(x)
    if not proj_out.is_contiguous():
        proj_out = proj_out.contiguous()
    return fused_swiglu(proj_out)

sa_transformer.GLU.forward = _glu_forward_fused_swiglu

print("Enabled sa_open_v6: KV cache + fused residual+LN + fused SwiGLU + fused RoPE (Triton)")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

DOWNSAMPLING_RATIO = 2048
LATENT_OFFLOAD_THRESHOLD = 256
PROFILE_DIT = False

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "..", "goldens")
LATENT_GOLDEN_PATH = os.path.join(GOLDENS_DIR, "latent_goldens.npy")
AUDIO_GOLDEN_PATH = os.path.join(GOLDENS_DIR, "audio_goldens.npy")
LATENT_RTL = 5e-1
LATENT_ATL = 4e-1
AUDIO_RTL = 5e-1
AUDIO_ATL = 3e-1

device = "cuda"

print("Loading model from HuggingFace...")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device).eval().requires_grad_(False)

sample_rate = model_config["sample_rate"]
sample_size = 1024 * DOWNSAMPLING_RATIO

latent_size = sample_size // DOWNSAMPLING_RATIO
duration = sample_size / sample_rate
offload = latent_size > LATENT_OFFLOAD_THRESHOLD

print(f"Sample rate: {sample_rate}, Sample size: {sample_size}, "
    f"Latent size: {latent_size}, Duration: {duration:.1f}s, Offload: {offload}")
print("Generating audio for prompt: 'The sound of hammer on wood'")

conditioning = build_conditioning("The sound of hammer on wood", duration)

seed = 42
setup_torch_backend(seed)
noise = create_noise(model, latent_size, device)

with torch.no_grad():
    torch.cuda.synchronize()
    t5_start = time.perf_counter()

    conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    torch.cuda.synchronize()
    t5_time = time.perf_counter() - t5_start

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    torch.cuda.synchronize()
    dit_start = time.perf_counter()

    dit_kwargs = build_dit_kwargs(device)

    if PROFILE_DIT:
        profile_dir = os.path.join(os.path.dirname(__file__), "dit_profile")
        os.makedirs(profile_dir, exist_ok=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
        ) as prof:
            with record_function("DiT_diffusion_5steps"):
                sampled = sample_k(
                    model.model, noise, None, 5,
                    **conditioning_inputs, **dit_kwargs,
                )
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
        print(f"\nProfile trace saved to {profile_dir}/")
        print("View with: tensorboard --logdir " + profile_dir)
    else:
        sampled = sample_k(
            model.model, noise, None, 100,
            **conditioning_inputs, **dit_kwargs,
        )

    torch.cuda.synchronize()
    dit_time = time.perf_counter() - dit_start
    latent_for_verify = sampled

    del noise, conditioning_tensors, conditioning_inputs
    clear_cross_attention_kv_cache(model.model)
    torch.cuda.empty_cache()

    if offload:
        print("Offloading DiT + T5 to CPU for VAE decode...")
        model.model.to("cpu")
        model.conditioner.to("cpu")
        del model.model
        del model.conditioner
        gc.collect()
        torch.cuda.empty_cache()

    sampled = sampled.to(next(model.pretransform.parameters()).dtype)

    torch.cuda.synchronize()
    vae_start = time.perf_counter()

    audio = model.pretransform.decode(sampled)

    torch.cuda.synchronize()
    vae_time = time.perf_counter() - vae_start

audio = normalize_audio(audio)

if os.path.exists(LATENT_GOLDEN_PATH) and os.path.exists(AUDIO_GOLDEN_PATH):
    latent_check = verify_golden_match(latent_for_verify, LATENT_GOLDEN_PATH, rtol=LATENT_RTL, atol=LATENT_ATL)
    print(f"Latent golden verification: match={latent_check['match']} "
        f"max_abs_diff={latent_check['max_abs_diff']:.6e} max_rel_diff={latent_check['max_rel_diff']:.6e}")

    audio_check = verify_golden_match(audio, AUDIO_GOLDEN_PATH, rtol=AUDIO_RTL, atol=AUDIO_ATL)
    print(f"Audio golden verification: match={audio_check['match']} "
        f"max_abs_diff={audio_check['max_abs_diff']:.6e} max_rel_diff={audio_check['max_rel_diff']:.6e}")
else:
    print(f"Latent golden not found at {LATENT_GOLDEN_PATH}, skipping verification")
    print(f"Audio golden not found at {AUDIO_GOLDEN_PATH}, skipping verification")

output_path = "audio_v6.wav"
sf.write(output_path, audio, sample_rate)

print_timing(t5_time, dit_time, vae_time, output_path)
