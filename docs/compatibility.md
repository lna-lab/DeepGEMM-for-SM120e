# SM120e Compatibility Notes

## Architecture Comparison

| Feature | SM90 (Hopper) | SM100 (Blackwell DC) | SM120e (Blackwell WS) |
|---------|:------------:|:-------------------:|:--------------------:|
| GPUs | H100, H800 | B200, B100 | RTX PRO 6000, RTX 5090 |
| Shared memory / SM | 228 KB | 168 KB | **49 KB** |
| `tcgen05.fence` | N/A | Supported | **Not supported** |
| TMA | Full | Full | Partial |
| FP8 tensor cores | Yes | Yes | Yes |
| FP4 tensor cores | No | Yes | **Yes** |
| NVFP4 quantization | No | Yes | **Yes** |
| CUDA minimum | 12.3 | 12.9 | 12.9+ |
| DeepGEMM support | Full | Full | **None (this project)** |

## Key Constraints for SM120e

### 1. Shared Memory (49 KB vs 168 KB)

DeepGEMM SM100 kernels assume 168 KB of shared memory per SM. SM120e has only 49 KB — roughly **29% of SM100**.

**Impact:**
- Tile sizes must be significantly reduced
- Double-buffering strategies may need simplification
- Some kernels may require multi-pass approaches

**Strategy:**
- Start with the smallest viable tile size
- Profile shared memory pressure per kernel
- Consider L2 cache as an alternative staging area

### 2. Barrier Instructions

SM100 uses `tcgen05.fence` for TMA (Tensor Memory Accelerator) synchronization. SM120e does not support this instruction.

**Impact:**
- Direct compilation of SM100 kernels fails
- TMA-dependent synchronization paths must be rewritten

**Strategy:**
- Identify all `tcgen05.fence` call sites in upstream kernels
- Replace with `__syncthreads()` or `cp.async.wait_group` equivalents
- Test for correctness — barrier replacements can introduce subtle race conditions

### 3. PTX Compatibility

NVIDIA recommends using `CUDA_FORCE_PTX_JIT=1` to verify kernel compatibility when targeting newer architectures.

```bash
CUDA_FORCE_PTX_JIT=1 python -c "import deep_gemm; deep_gemm.test_dense()"
```

## Upstream Status

- **Upstream target:** SM90 / SM100
- **This project target:** SM120e
- **Expected status:** Experimental
- **PTX/JIT verification:** Required
- **Unsupported kernels:** Tracked explicitly in this document

## Known Unsupported Kernels

| Kernel | SM100 Status | SM120e Status | Blocker |
|--------|:----------:|:------------:|---------|
| `fp8_gemm_nt` | ✓ | Planned | smem, barrier |
| `m_grouped_fp8_gemm_nt_masked` | ✓ | Planned | smem, barrier |
| `m_grouped_fp8_gemm_nt_contiguous` | ✓ | Planned | smem, barrier |
| `fp8_mqa_logits` | ✓ | Planned | `sm120_fp8_paged_mqa_logits.cuh` missing |
| `fp8_fp4_mega_moe` | ✓ | Planned | smem, barrier, comm overlap |

This table will be updated as kernels are ported and tested.
