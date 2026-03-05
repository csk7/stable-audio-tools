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

## TODO (DiT CUDA Kernel Fusions)

- [ ] Implement cross-attention KV caching: pre-compute K,V projections once per prompt and reuse across denoising steps/layers; bypass `to_kv` and `repeat_interleave` in cross-attention.
- [ ] Optimize attention path for long sequences (`N ~ 1024`): reduce dtype/layout churn around SDPA/FlashAttention, keep qkv in fp16/bf16 path consistently, and benchmark sliding-window option.
- [ ] Write Triton fused residual-add + LayerNorm kernel for long-sequence blocks (`x += sublayer_out; ln(x)`) on `(2, 1025, 1536)`-class shapes.
- [ ] Write Triton fused SwiGLU activation kernel (`chunk + SiLU + mul`) on post-GEMM tensors `(2, 1025, 12288) -> (2, 1025, 6144)`.
- [ ] Add Triton fused RoPE only if profiling still shows RoPE overhead after attention-path cleanup at `N ~ 1024`.
- [ ] Benchmark each change independently at latent `seq_len=1024`, and report per-step DiT latency and end-to-end latency with expected gain ranges.
