import torch
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention import SDPBackend, sdpa_kernel


def patch_attention_with_sdpa_flash(sa_transformer):
    def _flash_apply_attn(self, q, k, v, causal=None, **kwargs):
        if self.num_heads != self.kv_heads and k.shape[1] == self.kv_heads:
            heads_per_kv_head = self.num_heads // self.kv_heads
            k, v = map(lambda t: t.repeat_interleave(heads_per_kv_head, dim=1), (k, v))

        orig_dtype = q.dtype
        if orig_dtype != torch.float16 and orig_dtype != torch.bfloat16:
            q, k, v = q.half(), k.half(), v.half()
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=bool(causal))
        return out.to(orig_dtype)

    sa_transformer.Attention.apply_attn = _flash_apply_attn
    print("Patched attention to use PyTorch Flash Attention backend")


def setup_torch_backend(seed):
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False


def build_conditioning(prompt, duration):
    return [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": round(duration),
    }]


def build_dit_kwargs(device):
    return dict(
        cfg_scale=7,
        batch_cfg=True,
        rescale_cfg=True,
        device=device,
        sampler_type="dpmpp-3m-sde",
        sigma_min=0.03,
        sigma_max=1000,
    )


def create_noise(model, latent_size, device):
    return torch.randn([1, model.io_channels, latent_size], device=device)


def normalize_audio(audio):
    audio = rearrange(audio, "b d n -> d (b n)")
    return audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).cpu().numpy().T


def print_timing(t5_time, dit_time, vae_time, output_path):
    e2e_time = t5_time + dit_time + vae_time
    print("\n--- Timing ---")
    print(f"T5 (text encoding):  {t5_time:.2f}s")
    print(f"DiT (diffusion):     {dit_time:.2f}s")
    print(f"VAE (decode):        {vae_time:.2f}s")
    print(f"End-to-end:          {e2e_time:.2f}s")
    print(f"Audio saved to {output_path}")
