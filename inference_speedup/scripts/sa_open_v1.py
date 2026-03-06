import gc
import os
import time
import torch
import soundfile as sf
from torch.profiler import profile, record_function, ProfilerActivity

from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.sampling import sample_k
from stable_audio_tools.models import transformer as sa_transformer
from goldens_save import save_latent_golden, save_audio_golden
from tools.tools import (
    patch_attention_with_sdpa_flash,
    setup_torch_backend,
    build_conditioning,
    build_dit_kwargs,
    create_noise,
    normalize_audio,
    print_timing,
)

patch_attention_with_sdpa_flash(sa_transformer)

DOWNSAMPLING_RATIO = 2048
LATENT_OFFLOAD_THRESHOLD = 256
PROFILE_DIT = False
SAVE_GOLDENS = False
GOLDENS_DIR = os.path.join(os.path.dirname(__file__), "..", "goldens")
LATENT_GOLDEN_PATH = os.path.join(GOLDENS_DIR, "latent_goldens.npy")
AUDIO_GOLDEN_PATH = os.path.join(GOLDENS_DIR, "audio_goldens.npy")

device = "cuda"

print("Loading model from HuggingFace...")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device).eval().requires_grad_(False)

sample_rate = model_config["sample_rate"]
sample_size = 1024 * DOWNSAMPLING_RATIO

latent_size = sample_size // DOWNSAMPLING_RATIO
duration = sample_size / sample_rate
offload = latent_size > LATENT_OFFLOAD_THRESHOLD

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
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40))
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

audio = normalize_audio(audio)

if SAVE_GOLDENS:
    audio_golden_path = save_audio_golden(audio, GOLDENS_DIR)
    print(f"Saved audio golden to {audio_golden_path}")

output_path = "audio_v1.wav"
sf.write(output_path, audio, sample_rate)

print_timing(t5_time, dit_time, vae_time, output_path)
