# Upstream Diff Notes

## DeepGEMM Files Requiring SM120e Changes

Based on analysis of [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) and [Issue #236](https://github.com/deepseek-ai/DeepGEMM/issues/236).

### Architecture Detection
- `deep_gemm/utils/arch.py` — add SM120/SM120a to supported architectures
- JIT compilation targets — add `-arch=sm_120` / `-arch=sm_120a`

### Shared Memory
- All kernel files under `deep_gemm/impls/` — reduce tile sizes for 49 KB smem limit
- Double-buffering logic — may need single-buffer fallback

### Barrier Instructions
- Search pattern: `tcgen05.fence`, `tcgen05.commit`
- Replacement: `__syncthreads()`, `cp.async.wait_group`, or `bar.sync`
- Files: all `.cuh` kernel implementations

### Missing Files
- `deep_gemm/impls/sm120_fp8_paged_mqa_logits.cuh` — does not exist upstream
- Needs to be created from `sm100_fp8_paged_mqa_logits.cuh` with SM120e adaptations

### Scaling Factor Format
- SM90: FP32 scaling factors
- SM100: Packed UE8M0 format
- SM120e: **TBD** — verify which format is required

## Tracking Changes

All SM120e-specific changes should:
1. Be clearly marked with `// SM120e:` comments
2. Use `#ifdef` guards where possible for upstream compatibility
3. Be documented in this file as they are made
