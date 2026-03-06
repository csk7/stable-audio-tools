import gc
import os
import time
import torch
import soundfile as sf
from torch.profiler import profile, record_function, ProfilerActivity

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.sampling import sample_k
from stable_audio_tools.models import transformer as sa_transformer
from tools.tools import patch_attention_with_sdpa_flash, setup_torch_backend, build_conditioning
from tools.tools import build_dit_kwargs, create_noise, normalize_audio, print_timing

_orig_attention_forward = sa_transformer.Attention.forward

def _attention_forward_with_cross_kv_cache(self, x, context=None, rotary_pos_emb=None, causal=None, flex_attention_block_mask=None, flex_attention_score_mod=None, flash_attn_sliding_window=None ):
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
print("Enabled cross-attention KV caching in sa_open_v3")


DOWNSAMPLING_RATIO = 2048
LATENT_OFFLOAD_THRESHOLD = 256
PROFILE_DIT = False

device = "cuda"

print("Loading model from HuggingFace...")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device).eval().requires_grad_(False)

sample_rate = model_config["sample_rate"]
sample_size = 1024 * DOWNSAMPLING_RATIO

latent_size = sample_size // DOWNSAMPLING_RATIO
duration = sample_size / sample_rate
offload = latent_size > LATENT_OFFLOAD_THRESHOLD

print(
    f"Sample rate: {sample_rate}, Sample size: {sample_size}, "
    f"Latent size: {latent_size}, Duration: {duration:.1f}s, Offload: {offload}"
)
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

output_path = "audio_v3.wav"
sf.write(output_path, audio, sample_rate)

print_timing(t5_time, dit_time, vae_time, output_path)
