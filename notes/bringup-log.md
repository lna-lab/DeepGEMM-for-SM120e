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
