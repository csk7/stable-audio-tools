# Stable Audio Open – Inference Speedup

## Setup

- **Model**: `stabilityai/stable-audio-open-1.0`
- **GPU**: NVIDIA GeForce RTX 3070 (8 GB VRAM)
- **Attention**: PyTorch SDPA Flash Attention backend
- **Precision**: float32 (TF32 disabled)
- **Sampler**: dpmpp-3m-sde, 100 steps, cfg_scale=7

## Baseline Timing (latent size 1024, ~47.6s audio)

Offloading DiT + T5 to CPU was required before VAE decode to fit in 8 GB VRAM.
Offloading time is **excluded** from the measurements below.

| Version | Run                                   | T5    | DiT    | VAE   | End-to-end |
|:--------|:--------------------------------------|------:|-------:|------:|-----------:|
| v1      | Baseline                              | 0.36s | 42.98s | 1.22s | 44.55s     |
| v2      | v1 + `torch.compile`                  | 0.01s | 40.23s | 1.22s | 41.46s     |
| v3      | v1 + Cross-attention KV cache         | 0.36s | 42.49s | 1.22s | 44.07s     |
| v4      | v3 + Fused residual-add + LayerNorm   | 0.36s | 42.00s | 1.22s | 43.58s     |
| v5      | v4 + Fused SwiGLU                     | 0.36s | 41.18s | 1.22s | 42.76s     |
| v6      | v5 + Fused RoPE                       | 0.36s | 40.34s | 1.22s | 42.02s     |

## DiT Inference Speed Up

- v3. Implement cross-attention KV caching: pre-compute K,V projections once per prompt and reuse across denoising steps/layers; bypass `to_kv` and `repeat_interleave` in cross-attention.
- v4. Write Triton fused residual-add + LayerNorm kernel for long-sequence blocks (`x += sublayer_out; ln(x)`) on `(2, 1025, 1536)`-class shapes.
- v5. Write Triton fused SwiGLU activation kernel (`chunk + SiLU + mul`) on post-GEMM tensors `(2, 1025, 12288) -> (2, 1025, 6144)`.
- v6. Add Triton fused RoPE `N ~ 1024`.
