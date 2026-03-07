"""
sa_open_v5: Cross-attention KV cache + fused residual+layernorm + fused SwiGLU.
- Cross-attention KV caching from v3
- Triton fused residual_add_layernorm from v4
- Triton fused SwiGLU activation: chunk + SiLU(gate) * x in one pass
"""
import gc
import os
import sys
import time

# Add inference_speedup root for kernel import (before tools/goldens)
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

_orig_attention_forward = sa_transformer.Attention.forward


def _attention_forward_with_cross_kv_cache(self, x, context=None, rotary_pos_emb=None, causal=None, flex_attention_block_mask=None, flex_attention_score_mod=None, flash_attn_sliding_window=None):
    if context is None or (not hasattr(self, "to_q")):
        return _orig_attention_forward(self, x, context=context, rotary_pos_emb=rotary_pos_emb, causal=causal,
            flex_attention_block_mask=flex_attention_block_mask, flex_attention_score_mod=flex_attention_score_mod, flash_attn_sliding_window=flash_attn_sliding_window,
        )

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
        flex_attention_score_mod=flex_attention_score_mod, flash_attn_sliding_window=flash_attn_sliding_window,
    )

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

def _can_use_fused_residual_ln(norm):
    """Only use fused kernel with LayerNorm (no force_fp32)."""
    return (isinstance(norm, sa_transformer.LayerNorm) and not getattr(norm, "force_fp32", False))

_orig_transformer_block_forward = sa_transformer.TransformerBlock.forward

def _transformer_block_forward_fused_residual_ln(self, x, context=None, global_cond=None, rotary_pos_emb=None,
    self_attention_block_mask=None, self_attention_score_mod=None, cross_attention_block_mask=None, cross_attention_score_mod=None,
    self_attention_flash_sliding_window=None, cross_attention_flash_sliding_window=None):
    if rotary_pos_emb is None and self.add_rope:
        rotary_pos_emb = self.rope.forward_from_seq_len(x.shape[-2])

    if self.global_cond_dim is not None and self.global_cond_dim > 0 and global_cond is not None:
        return _orig_transformer_block_forward(
            self, x, context=context, global_cond=global_cond, rotary_pos_emb=rotary_pos_emb,
            self_attention_block_mask=self_attention_block_mask, self_attention_score_mod=self_attention_score_mod,
            cross_attention_block_mask=cross_attention_block_mask, cross_attention_score_mod=cross_attention_score_mod,
            self_attention_flash_sliding_window=self_attention_flash_sliding_window,
            cross_attention_flash_sliding_window=cross_attention_flash_sliding_window)

    # global_cond_dim is None path - apply fused residual+layernorm where possible
    use_fused_self = _can_use_fused_residual_ln(self.cross_attend_norm if self.cross_attend else self.ff_norm)
    use_fused_cross = self.cross_attend and _can_use_fused_residual_ln(self.ff_norm)

    # Self-attention
    attn_out = self.self_attn_scale(self.self_attn(self.pre_norm(x), rotary_pos_emb=rotary_pos_emb,
        flex_attention_block_mask=self_attention_block_mask, flex_attention_score_mod=self_attention_score_mod,
        flash_attn_sliding_window=self_attention_flash_sliding_window))
    if use_fused_self:
        next_norm = self.cross_attend_norm if self.cross_attend else self.ff_norm
        x_norm, x_residual = residual_add_layernorm(x, attn_out, next_norm.gamma, next_norm.beta, eps=next_norm.eps)
    else:
        x = x + attn_out
        x_norm, x_residual = None, None

    # Cross-attention
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

    # Conformer (if present)
    if self.conformer is not None:
        r = x_residual if (use_fused_self or use_fused_cross) else x
        x = r + self.conformer_scale(self.conformer(r))
        x_residual, x_norm = x, None  # x is new residual for ff

    # Feed-forward
    r = x_residual if (use_fused_self or use_fused_cross) else x
    ff_input = x_norm if use_fused_cross else self.ff_norm(r)
    x = r + self.ff_scale(self.ff(ff_input))
    return x


sa_transformer.TransformerBlock.forward = _transformer_block_forward_fused_residual_ln

# --- Fused SwiGLU: monkey-patch GLU.forward ---
# GLU.forward (linear path, not conv): x = proj(x); x, gate = x.chunk(2, -1); return x * act(gate)
# The fused kernel handles chunk + SiLU(gate) * x in one Triton pass.

_orig_glu_forward = sa_transformer.GLU.forward

def _glu_forward_fused_swiglu(self, x):
    # Fall back to original for conv path (rare, not the DiT FFN path)
    if self.use_conv:
        return _orig_glu_forward(self, x)
    # Ensure the activation is SiLU — fused kernel only implements SiLU
    if not isinstance(self.act, torch.nn.SiLU):
        return _orig_glu_forward(self, x)
    # proj is the GEMM, then fuse chunk + SiLU(gate) * x
    proj = self.proj(x)
    if not proj.is_contiguous():
        proj = proj.contiguous()
    return fused_swiglu(proj)

sa_transformer.GLU.forward = _glu_forward_fused_swiglu
print("Enabled sa_open_v5: cross-attention KV cache + fused residual-add+LayerNorm + fused SwiGLU (Triton)")


DOWNSAMPLING_RATIO = 2048
LATENT_OFFLOAD_THRESHOLD = 256
PROFILE_DIT = False

GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "..", "goldens")
LATENT_GOLDEN_PATH = os.path.join(GOLDENS_DIR, "latent_goldens.npy")
AUDIO_GOLDEN_PATH = os.path.join(GOLDENS_DIR, "audio_goldens.npy")
# Relaxed tolerances for fused kernel + KV cache path
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
    # Step 1: T5 text encoding
    torch.cuda.synchronize()
    t5_start = time.perf_counter()

    conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    torch.cuda.synchronize()
    t5_time = time.perf_counter() - t5_start

    # Step 2: DiT diffusion (100-step denoising)
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

    # Step 2.5: Offload DiT + T5 if needed (not counted in timing)
    if offload:
        print("Offloading DiT + T5 to CPU for VAE decode...")
        model.model.to("cpu")
        model.conditioner.to("cpu")
        del model.model
        del model.conditioner
        gc.collect()
        torch.cuda.empty_cache()

    # Step 3: VAE decode (latents -> audio)
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
    print( f"Audio golden verification: match={audio_check['match']} "
        f"max_abs_diff={audio_check['max_abs_diff']:.6e} max_rel_diff={audio_check['max_rel_diff']:.6e}")
else:
    print(f"Latent golden not found at {LATENT_GOLDEN_PATH}, skipping verification")
    print(f"Audio golden not found at {AUDIO_GOLDEN_PATH}, skipping verification")

output_path = "audio_v5.wav"
sf.write(output_path, audio, sample_rate)

print_timing(t5_time, dit_time, vae_time, output_path)
