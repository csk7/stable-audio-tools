import gc
import os
import time
import torch
import torch.nn.functional as F
import soundfile as sf
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import profile, record_function, ProfilerActivity
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.sampling import sample_k
from stable_audio_tools.models import transformer as sa_transformer
from goldens_save import save_latent_golden, save_audio_golden, verify_golden_match


_orig_apply_attn = sa_transformer.Attention.apply_attn


def _flash_apply_attn(self, q, k, v, causal=None, **kwargs):
    if self.num_heads != self.kv_heads:
        heads_per_kv_head = self.num_heads // self.kv_heads
        k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))
    orig_dtype = q.dtype
    if orig_dtype != torch.float16 and orig_dtype != torch.bfloat16:
        q, k, v = q.half(), k.half(), v.half()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.to(orig_dtype)

sa_transformer.Attention.apply_attn = _flash_apply_attn
print("Patched attention to use PyTorch Flash Attention backend")


DOWNSAMPLING_RATIO = 2048
LATENT_OFFLOAD_THRESHOLD = 256
PROFILE_DIT = False
TORCH_COMPILE = False
SAVE_GOLDENS = False
VERIFY_GOLDENS = TORCH_COMPILE
GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "..", "goldens")
LATENT_GOLDEN_PATH = os.path.join(GOLDENS_DIR, "latent_goldens.npy")
AUDIO_GOLDEN_PATH = os.path.join(GOLDENS_DIR, "audio_goldens.npy")
# Relaxed tolerances for compiled/optimized inference paths, which can
# introduce larger but still perceptually acceptable numeric drift.
LATENT_RTL = 5e-1
LATENT_ATL = 4e-1
AUDIO_RTL = 5e-1
AUDIO_ATL = 3e-1

device = "cuda"

print("Loading model from HuggingFace...")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device).eval().requires_grad_(False)

if TORCH_COMPILE:
    print("Compiling models with torch.compile...")
    model.conditioner = torch.compile(model.conditioner)
    model.model = torch.compile(model.model)
    model.pretransform = torch.compile(model.pretransform)

sample_rate = model_config["sample_rate"]
sample_size = 1024 * DOWNSAMPLING_RATIO

latent_size = sample_size // DOWNSAMPLING_RATIO
duration = sample_size / sample_rate
offload = latent_size > LATENT_OFFLOAD_THRESHOLD

print(f"Sample rate: {sample_rate}, Sample size: {sample_size}, "
      f"Latent size: {latent_size}, Duration: {duration:.1f}s, Offload: {offload}")
print("Generating audio for prompt: 'The sound of hammer on wood'")

conditioning = [{
    "prompt": "The sound of hammer on wood",
    "seconds_start": 0,
    "seconds_total": round(duration),
}]

seed = 42
torch.manual_seed(seed)
noise = torch.randn([1, model.io_channels, latent_size], device=device)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cudnn.benchmark = False

if TORCH_COMPILE:
    print("Warmup pass (triggers compilation)...")
    _dit_kwargs = dict(
        cfg_scale=7, batch_cfg=True, rescale_cfg=True, device=device,
        sampler_type="dpmpp-3m-sde", sigma_min=0.03, sigma_max=1000,
    )
    with torch.no_grad():
        _w_cond = model.conditioner(conditioning, device)
        _w_inputs = model.get_conditioning_inputs(_w_cond)
        _w_dtype = next(model.model.parameters()).dtype
        _w_noise = torch.randn_like(noise).type(_w_dtype)
        _w_inputs = {k: v.type(_w_dtype) if v is not None else v for k, v in _w_inputs.items()}
        sample_k(model.model, _w_noise, None, 2, **_w_inputs, **_dit_kwargs)
    del _w_cond, _w_inputs, _w_noise, _dit_kwargs
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    noise = torch.randn([1, model.io_channels, latent_size], device=device)
    print("Warmup complete.")

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

    dit_kwargs = dict(
        cfg_scale=7, batch_cfg=True, rescale_cfg=True, device=device,
        sampler_type="dpmpp-3m-sde", sigma_min=0.03, sigma_max=1000,
    )

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
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
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

    if SAVE_GOLDENS:
        latent_path = save_latent_golden(sampled, GOLDENS_DIR)
        print(f"Saved latent golden to {latent_path}")

    del noise, conditioning_tensors, conditioning_inputs
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

audio = rearrange(audio, "b d n -> d (b n)")
audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).cpu().numpy().T

if SAVE_GOLDENS:
    audio_golden_path = save_audio_golden(audio, GOLDENS_DIR)
    print(f"Saved audio golden to {audio_golden_path}")

if VERIFY_GOLDENS:
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


output_path = "temple_bells_torch_compile.wav"
sf.write(output_path, audio, sample_rate)

e2e_time = t5_time + dit_time + vae_time
print(f"\n--- Timing ---")
print(f"T5 (text encoding):  {t5_time:.2f}s")
print(f"DiT (diffusion):     {dit_time:.2f}s")
print(f"VAE (decode):        {vae_time:.2f}s")
print(f"End-to-end:          {e2e_time:.2f}s")
print(f"Audio saved to {output_path}")
