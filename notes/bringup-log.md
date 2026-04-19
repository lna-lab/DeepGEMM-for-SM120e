# Bring-up Log

## 2026-04-19 — Repository initialization

- Created project structure based on upstream DeepGEMM analysis
- Identified SM120e blockers: 49 KB shared memory (vs 168 KB), missing `tcgen05.fence`
- Documented architecture comparison in `docs/compatibility.md`
- Proposed tile sizes: M=64, N=64, K=32 (20 KB) or K=64 (24 KB) for 49 KB smem
- Prior art review: Issue #236, createthis fork (barrier issue at 46 KB smem)

### Key References
- Upstream: https://github.com/deepseek-ai/DeepGEMM (public-release-260416)
- SM120 issue: https://github.com/deepseek-ai/DeepGEMM/issues/236
- Attempted port: https://github.com/createthis/DeepGEMM/pull/1

### Hardware Available
- 7x RTX PRO 6000 Blackwell (SM120, 96 GB each)
- CUDA 13.0, Driver 580.126.09
- GPU 0 designated for kernel development/testing

### Next Steps
1. Run `scripts/bootstrap.sh` to clone upstream and set up dev env
2. Run `scripts/test_sm120e_env.sh` to verify environment
3. Study `deep_gemm/impls/` for shared memory and barrier patterns
4. Start with `fp8_gemm_nt` kernel port

## 2026-04-19 — Environment verification complete

### test_sm120e_env.sh results
```
GPU:              RTX PRO 6000 Blackwell (SM120, compute 12.0)
Shared memory:    49,152 bytes (48 KB)
VRAM:             102 GB
PyTorch:          2.11.0+cu130
CUTLASS:          found
FP8 E4M3:        supported
nvcc:             not found (CUDA toolkit not installed system-wide)
Driver:           580.126.09
```

### GPU allocation
- GPU 0: dedicated to DeepGEMM development (96 GB free)
- GPU 1-6: DFlash hidden state collection running (6 workers, ~1,500 samples collected)

### Observations
- `shared_memory_per_block = 49,152` confirms the 49 KB constraint documented in compatibility.md
- FP8 E4M3 tensor creation works — tensor core path should be functional
- CUTLASS is available — can use as reference or fallback
- nvcc missing — DeepGEMM JIT uses NVRTC as alternative, may be sufficient

### Issues found
- PyTorch API: `props.max_shared_memory_per_block` → `props.shared_memory_per_block` (fixed)
- PyTorch API: `props.total_mem` → `props.total_memory` (fixed)

### Next
- Clone upstream DeepGEMM and study kernel code
- Check if NVRTC path works without nvcc
- Identify all tcgen05.fence locations in upstream
