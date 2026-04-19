# SM120e Architecture Notes

## GPU Specifications (RTX PRO 6000 Blackwell Workstation Edition)

| Spec | Value |
|------|-------|
| Architecture | Blackwell (SM120) |
| SMs | 170 |
| CUDA Cores | 21,760 |
| Tensor Cores | 680 (5th gen) |
| Memory | 96 GB GDDR7 ECC |
| Memory Bandwidth | 1,536 GB/s |
| FP8 Tensor TFLOPS | TBD (theoretical) |
| FP4 Tensor TFLOPS | TBD (theoretical) |
| Shared Memory / SM | 49,152 bytes (48 KB) |
| L2 Cache | TBD |
| TDP | 350W |
| PCIe | Gen5 x16 |
| MIG Support | Up to 4 instances (2g.48gb profile available) |

## SM120 vs SM100 Instruction Differences

### Supported on SM120
- Standard CUDA intrinsics
- `cp.async` for async memory copy
- `mma` (matrix multiply-accumulate) instructions for FP8/FP4
- Tensor core operations for E4M3, E5M2, FP4

### NOT Supported on SM120
- `tcgen05.fence` — TMA barrier fence
- `tcgen05.commit` — TMA commit (verification needed)
- Large shared memory configurations (>49 KB)

## Shared Memory Tiling Analysis

### DeepGEMM SM100 Typical Tile Configuration
```
Tile M=128, N=128, K=64
A tile: 128 × 64 × sizeof(fp8) = 8,192 bytes
B tile: 64 × 128 × sizeof(fp8) = 8,192 bytes
Accumulator: 128 × 128 × sizeof(fp32) = 65,536 bytes
Total: ~82 KB (double-buffered: ~164 KB)
→ Fits in SM100's 168 KB ✓
→ Does NOT fit in SM120's 49 KB ✗
```

### Proposed SM120e Tile Configuration
```
Option A: Tile M=64, N=64, K=32
A tile: 64 × 32 × sizeof(fp8) = 2,048 bytes
B tile: 32 × 64 × sizeof(fp8) = 2,048 bytes
Accumulator: 64 × 64 × sizeof(fp32) = 16,384 bytes
Total: ~20 KB (double-buffered: ~40 KB)
→ Fits in SM120's 49 KB ✓

Option B: Tile M=64, N=64, K=64
A tile: 64 × 64 × sizeof(fp8) = 4,096 bytes
B tile: 64 × 64 × sizeof(fp8) = 4,096 bytes
Accumulator: 64 × 64 × sizeof(fp32) = 16,384 bytes
Total: ~24 KB (double-buffered: ~48 KB)
→ Tight fit in SM120's 49 KB — verify with padding ⚠️
```

### Trade-offs
- Smaller tiles = more iterations per output tile = more global memory traffic
- But SM120e has high memory bandwidth (1,536 GB/s)
- Optimal tile size must be empirically determined via benchmarking

## Useful Commands

```bash
# Check GPU architecture
nvidia-smi --query-gpu=index,name,compute_cap --format=csv,noheader

# Check SM count
nvidia-smi --query-gpu=index,name,clocks.max.sm --format=csv,noheader

# Force PTX JIT (compatibility test)
CUDA_FORCE_PTX_JIT=1 python test_kernel.py

# Check available shared memory
python -c "import torch; print(torch.cuda.get_device_properties(0).total_global_mem)"
```
