# DeepGEMM-for-SM120e

Experimental SM120e port and bring-up project for [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) on NVIDIA Blackwell workstation/consumer GPUs.

## Overview

[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) delivers up to 1550 TFLOPS on Hopper (SM90) and supports Blackwell data center GPUs (SM100), but **does not support SM120/SM120a** — the architecture used in RTX PRO 6000, RTX 5090, RTX 5080, and other Blackwell workstation/consumer GPUs.

This repository aims to close that gap by:

- bringing up the build and JIT pipeline on SM120e
- adding missing architecture detection and kernel dispatch paths
- validating correctness first, then performance
- prioritizing dense GEMM → grouped GEMM → MQA → Mega MoE in that order
- keeping the codebase reviewable and upstream-aware

This is **not** a full official DeepGEMM replacement.
The immediate goal is to achieve a practical and testable SM120e path for real-world inference workloads.

## Why SM120e Needs a Dedicated Port

| | SM100 (B200) | SM120e (RTX PRO 6000) |
|---|---|---|
| Shared memory | 168 KB per SM | **49 KB per SM** |
| `tcgen05.fence` | Supported | **Not supported** |
| TMA barrier | Full support | **Requires alternative sync** |
| DeepGEMM status | Works | **Fails to compile** |

The 3.4x smaller shared memory and missing barrier instructions mean SM120e kernels must be **redesigned, not just recompiled**.

## Project Goals

### Phase 1 — Bring-up
- Detect SM120e correctly in architecture checks
- Make JIT / compile targets work on Blackwell workstation GPUs
- Run minimal dense GEMM kernels successfully
- Verify correctness against PyTorch baselines

### Phase 2 — Core Coverage
- Enable grouped GEMM paths relevant to MoE workloads
- Establish test coverage for FP8 / FP4 / BF16 code paths
- Document unsupported / partially supported kernels clearly

### Phase 3 — Practical Inference Use
- Target DeepSeek-style and Qwen3.x MoE inference workloads
- Evaluate integration with vLLM and llama.cpp inference backends
- Prepare hooks for future SM120e kernel optimizations

## Non-Goals (for now)

- Full parity with upstream DeepGEMM on day one
- Premature low-level micro-optimization before correctness
- Immediate support for every kernel family at once
- Claiming official upstream compatibility

## Development Principles

1. Correctness before speed
2. Smallest working kernel before large refactors
3. Architecture-specific changes must be isolated and documented
4. Benchmark only after correctness is stable
5. Keep the codebase easy to review

## Work Order

| # | Task | Kernel Family | Status |
|---|------|---------------|--------|
| 1 | Architecture detection | — | Not started |
| 2 | Build / JIT pipeline on SM120e | — | Not started |
| 3 | Dense GEMM kernel (`fp8_gemm_nt`) | Dense | Not started |
| 4 | Grouped GEMM for MoE | MoE | Not started |
| 5 | MQA scoring kernel | Attention | Not started |
| 6 | Mega MoE / fused paths | MoE + Comm | Not started |

## Repository Layout

```text
.
├── README.md
├── docs/
│   ├── roadmap.md
│   ├── sm120e-notes.md
│   ├── compatibility.md
│   └── benchmark-plan.md
├── patches/
│   ├── upstream-diff-notes.md
│   └── jit-sm120e-notes.md
├── scripts/
│   ├── bootstrap.sh
│   ├── test_sm120e_env.sh
│   └── run_smoke_tests.sh
├── tests/
│   ├── test_dense_sm120e.py
│   ├── test_grouped_sm120e.py
│   └── test_correctness_vs_torch.py
└── notes/
    └── bringup-log.md
```

## First Milestone

The first milestone is considered complete when all of the following are true:

- [ ] Repository builds on an SM120e machine
- [ ] At least one dense GEMM path runs successfully
- [ ] Outputs match a trusted reference implementation
- [ ] Test and benchmark procedures are documented
- [ ] Known limitations are written down clearly

## Hardware

Development and testing on:
- **7x NVIDIA RTX PRO 6000 Blackwell Workstation Edition** (96 GB GDDR7 each, SM120)
- CUDA 13.0, Driver 580.126.09

## Prior Art

- [DeepGEMM Issue #236](https://github.com/deepseek-ai/DeepGEMM/issues/236) — SM120 feature request, identifies shared memory and barrier blockers
- [createthis/DeepGEMM#1](https://github.com/createthis/DeepGEMM/pull/1) — Attempted port, reduced smem to 46 KB but hit barrier issues
- [DeepGEMM public-release-260416](https://github.com/deepseek-ai/DeepGEMM/tree/public-release-260416) — Latest upstream with FP8xFP4, Mega MoE, PDL

## Status

Current status: **bootstrap / documentation stage**.

## License

TBD — pending alignment with [DeepGEMM's MIT license](https://github.com/deepseek-ai/DeepGEMM/blob/main/LICENSE).

## Acknowledgements

- Original library: [deepseek-ai/DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- SM120e port: [Lna-Lab](https://github.com/lna-lab)
