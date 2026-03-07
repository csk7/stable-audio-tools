# Fused Residual-Add + LayerNorm Kernel Improvements

This document describes the optimizations applied to the Triton fused residual-add + LayerNorm kernel used in the DiT transformer blocks.

## Purpose

The kernel fuses two operations that were originally separate:

1. **Residual add**: `out_sum = residual + sublayer_out`
2. **LayerNorm**: `out_norm = layernorm(out_sum)`

It returns both `(out_norm, out_sum)` so the next sublayer can add the residual. Used after self-attention and cross-attention in `TransformerBlock`.

## Original Performance

- **Initial fused kernel**: ~234 µs per call
- **Separate add + layernorm**: ~177 µs per call

The fused kernel was *slower* than the separate path due to redundant memory traffic and suboptimal configuration.

## Improvements

### 1. Single-Pass Mean and Variance

**Before**: Two separate passes over the data for statistics:
- Pass 1: Compute mean (load R, S)
- Pass 2: Compute variance (load R, S again)
- Pass 3: Output (load R, S, W, B)

**After**: One pass for statistics using the identity `var = E[x²] - E[x]²`:
- Pass 1: Compute `sum(x)` and `sum(x²)` in one loop (load R, S once)
- Pass 2: Output (load R, S, W, B)

**Impact**: Reduced global memory reads of Residual and SublayerOut from 3 passes to 2 (~33% fewer loads for the stats phase).

### 2. Larger BLOCK_SIZE

**Before**: `BLOCK_SIZE = min(1024, next_power_of_2(N))` — for N=1536, BLOCK_SIZE=1024 → 2 loop iterations per pass.

**After**: `BLOCK_SIZE = min(2048, next_power_of_2(N))` — for N=1536, BLOCK_SIZE=2048 → 1 loop iteration per pass.

**Impact**: Fewer loop iterations, less loop overhead. For typical DiT shapes (N=1536), each pass now does a single block with masking.

### 3. FP16 Output for OutNorm

**Before**: OutNorm allocated as float32, kernel stored float32, then `.to(residual.dtype)` conversion at the end.

**After**: OutNorm allocated in input dtype (float16 when model uses fp16). Kernel converts and stores directly in float16 via `STORE_FP16` constexpr.

**Impact**: Halved store bandwidth for OutNorm; eliminated post-kernel conversion.

### 4. More Warps

**Before**: `num_warps = min(max(BLOCK_SIZE // 256, 1), 8)` — typically 4 for BLOCK_SIZE=1024.

**After**: `num_warps = 8`

**Impact**: Better latency hiding for memory-bound workload.

### 5. Multi-Row Processing

**Before**: One program per row. Each program loaded Weight and Bias independently.

**After**: Each program processes 2 rows (`ROWS_PER_PROGRAM=2`). Weight and Bias are loaded once per block and reused for both rows.

**Impact**: ~50% reduction in Weight/Bias memory reads. Fewer kernel launches (M/2 programs instead of M).

### 6. num_stages Support

Added `num_stages` (default 2) for software pipelining. In benchmarks, no measurable difference was observed—likely because for N=1536 with BLOCK_SIZE=2048 there is only one loop iteration, leaving nothing to pipeline. Configurable via `RESIDUAL_LN_NUM_STAGES`.

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `USE_PYTORCH_FUSED` | `0` | Set to `1` to fallback to PyTorch (add + F.layer_norm) |
| `RESIDUAL_LN_MULTI_ROW` | `1` | Set to `0` for single-row kernel (one program per row) |
| `RESIDUAL_LN_NUM_STAGES` | `2` | Triton pipelining stages (1, 2, or 3) |

## Benchmark Results (Typical Shape: M=1200, N=1536)

| Configuration | Time (µs) |
|---------------|-----------|
| Original fused kernel | ~234 |
| Separate add + layernorm | ~177 |
| Optimized fused (single-row) | ~41 |
| Optimized fused (multi-row, default) | ~47 |

The optimized kernel is ~5× faster than the original fused version and ~2–4× faster than the separate path in isolated benchmarks. Actual gains in the full model depend on GPU contention and memory pressure.

## Kernel Variants

- **Single-row** (`_residual_add_layernorm_fwd_kernel_single_row`): One program per row. Use when `RESIDUAL_LN_MULTI_ROW=0`. Slightly faster in isolation for some shapes.
- **Multi-row** (`_residual_add_layernorm_fwd_kernel`): Two rows per program, Weight/Bias reuse. Default. Can be faster under load due to reduced memory traffic.
