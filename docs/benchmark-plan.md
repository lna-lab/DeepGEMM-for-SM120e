# Benchmark Plan

## Baseline Measurements

Before any kernel porting, establish baselines on SM120e using existing libraries.

### cuBLAS Baseline
```python
import torch
A = torch.randn(M, K, dtype=torch.float8_e4m3fn, device='cuda')
B = torch.randn(K, N, dtype=torch.float8_e4m3fn, device='cuda')
# Warm up + time cuBLAS GEMM via torch.mm
```

### CUTLASS Baseline
- Use CUTLASS 4.0+ SM120 kernels if available

### Target Shapes (LLM Inference)

| Shape | M | N | K | Use Case |
|-------|---|---|---|----------|
| Small decode | 1-8 | 2048 | 2048 | Single token generation |
| Batch decode | 32-128 | 2048 | 2048 | Continuous batching |
| Prefill | 512-4096 | 2048 | 2048 | Prompt processing |
| MoE expert | 1-64 | 512 | 2048 | Qwen3.6 (256 experts, dim=512) |
| Large prefill | 8192 | 2048 | 2048 | Long context |

### Metrics
- TFLOPS (FP8)
- Utilization % vs theoretical peak
- Latency (μs)
- Memory bandwidth utilization

## Phase 1 Benchmark
- Dense `fp8_gemm_nt` vs cuBLAS on shapes above
- Goal: ≥ 80% of cuBLAS performance

## Phase 2 Benchmark
- Grouped GEMM vs naive batched GEMM
- MoE dispatch latency comparison

## Phase 3 Benchmark
- End-to-end model inference: tok/s with and without DeepGEMM-SM120e
- Target models: Qwen3.6-35B-A3B, GLM-5.1
