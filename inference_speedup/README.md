# Stable Audio Open – Inference Speedup

## Setup

- **Model**: `stabilityai/stable-audio-open-1.0`
- **GPU**: NVIDIA GeForce RTX 4060 (8 GB VRAM)
- **Attention**: PyTorch SDPA Flash Attention backend (monkey-patched)
- **Precision**: float32 (TF32 disabled)
- **Sampler**: dpmpp-3m-sde, 100 steps, cfg_scale=7

## Baseline Timing (latent size 1024, ~47.6s audio)

Offloading DiT + T5 to CPU was required before VAE decode to fit in 8 GB VRAM.
Offloading time is **excluded** from the measurements below.

| Run            | T5     | DiT    | VAE   | End-to-end |
|----------------|--------|--------|-------|------------|
| Baseline       | 0.36s  | 42.58s | 1.22s | 44.15s     |
| torch.compile  | 0.01s  | 40.13s | 1.22s | 41.36s     |
