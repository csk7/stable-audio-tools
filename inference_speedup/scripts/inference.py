import gc
import torch
import soundfile as sf
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond

DOWNSAMPLING_RATIO = 2048
LATENT_OFFLOAD_THRESHOLD = 256

device = "cuda"

print("Loading model from HuggingFace...")
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
model = model.to(device).eval().requires_grad_(False)

sample_rate = model_config["sample_rate"]
#sample_size = model_config["sample_size"]
#sample_size = 256 * 2048  # 256 latent frames × 2048 downsampling ratio = 524288 samples ≈ 11.9s
sample_size = 256 * DOWNSAMPLING_RATIO

latent_size = sample_size // DOWNSAMPLING_RATIO
duration = sample_size / sample_rate
offload = latent_size > LATENT_OFFLOAD_THRESHOLD

print(f"Sample rate: {sample_rate}, Sample size: {sample_size}, "
      f"Latent size: {latent_size}, Duration: {duration:.1f}s, Offload: {offload}")
print("Generating audio for prompt: 'The sound of temple bells'")

conditioning = [{
    "prompt": "The sound of temple bells",
    "seconds_start": 0,
    "seconds_total": round(duration),
}]

if offload:
    # Step 1: Run diffusion, return latents only (skip VAE decode)
    with torch.no_grad():
        audio = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            batch_size=1,
            sample_size=sample_size,
            seed=42,
            device=device,
            sampler_type="dpmpp-3m-sde",
            sigma_min=0.03,
            sigma_max=1000,
            return_latents=True,
        )

    # Step 2: Offload DiT + T5 to CPU to free VRAM for VAE decode
    print("Offloading DiT + T5 to CPU for VAE decode...")
    model.model.to("cpu")
    model.conditioner.to("cpu")
    del model.model
    del model.conditioner
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Decode latents -> audio using the VAE on GPU
    print("Decoding latents to audio...")
    with torch.no_grad():
        audio = audio.to(next(model.pretransform.parameters()).dtype)
        audio = model.pretransform.decode(audio)
else:
    # Step 1-3: Run diffusion + VAE decode end-to-end on GPU (fits in VRAM)
    with torch.no_grad():
        audio = generate_diffusion_cond(
            model,
            steps=100,
            cfg_scale=7,
            conditioning=conditioning,
            batch_size=1,
            sample_size=sample_size,
            seed=42,
            device=device,
            sampler_type="dpmpp-3m-sde",
            sigma_min=0.03,
            sigma_max=1000,
        )

audio = rearrange(audio, "b d n -> d (b n)")
audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).cpu().numpy().T

output_path = "temple_bells.wav"
sf.write(output_path, audio, sample_rate)
print(f"Audio saved to {output_path}")
