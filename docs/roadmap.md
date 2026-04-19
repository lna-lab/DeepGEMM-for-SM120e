# Roadmap

## Phase 1 — Bring-up (Current)

### 1.1 Architecture Detection
- [ ] Add SM120/SM120a to architecture checks in DeepGEMM JIT module
- [ ] Verify `nvcc` and `nvrtc` compilation targets for SM120
- [ ] Document PTX compatibility between SM100 and SM120

### 1.2 Build Pipeline
- [ ] `develop.sh` — make JIT module compile on SM120e
- [ ] `install.sh` — verify package installation
- [ ] NVRTC path — test runtime compilation on SM120e

### 1.3 First Kernel (Dense GEMM)
- [ ] Port `fp8_gemm_nt` with reduced shared memory (≤ 49 KB)
- [ ] Replace `tcgen05.fence` with SM120-compatible barrier
- [ ] Validate output against `torch.matmul` reference
- [ ] Measure TFLOPS on RTX PRO 6000

## Phase 2 — Core Coverage

### 2.1 Grouped GEMM for MoE
- [ ] `m_grouped_fp8_gemm_nt_masked` (decoding, variable expert loads)
- [ ] `m_grouped_fp8_gemm_nt_contiguous` (prefill, batch dispatch)
- [ ] Test with Qwen3.6 MoE (256 experts) weight shapes

### 2.2 Data Type Coverage
- [ ] FP8 (E4M3 / E5M2) — primary target
- [ ] FP4 (NVFP4) — for NVFP4 inference
- [ ] BF16 — fallback path
- [ ] FP8xFP4 mixed — for Mega MoE

### 2.3 Test Infrastructure
- [ ] Correctness suite (vs PyTorch, various shapes)
- [ ] Performance benchmark harness
- [ ] CI-friendly smoke tests

## Phase 3 — Practical Inference

### 3.1 Attention Kernels
- [ ] `fp8_mqa_logits` — standard MQA scoring
- [ ] `fp8_paged_mqa_logits` — paged KV cache scoring

### 3.2 Mega MoE
- [ ] Fused MoE with overlapped communication
- [ ] Multi-GPU validation

### 3.3 Integration
- [ ] vLLM backend hook (custom GEMM dispatch for SM120e)
- [ ] llama.cpp GGML operator override
- [ ] Benchmark: end-to-end inference speedup on real models
