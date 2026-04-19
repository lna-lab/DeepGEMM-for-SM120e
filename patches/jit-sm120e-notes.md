# JIT SM120e Notes

## DeepGEMM JIT System

DeepGEMM compiles CUDA kernels at runtime via a lightweight JIT module. This avoids requiring CUDA compilation at install time but means architecture support must be explicitly added.

## Changes Needed for SM120e

### 1. Architecture String
```python
# Current (upstream)
supported_archs = ['sm_90', 'sm_90a', 'sm_100', 'sm_100a']

# With SM120e
supported_archs = ['sm_90', 'sm_90a', 'sm_100', 'sm_100a', 'sm_120', 'sm_120a']
```

### 2. Compute Capability Detection
```python
# PyTorch way
cc = torch.cuda.get_device_capability()  # Returns (12, 0) for SM120
```

### 3. NVCC Flags
```bash
# SM120e compilation
nvcc -arch=sm_120a -std=c++20 kernel.cu

# NVRTC (runtime compilation)
# Add "-arch=compute_120" to NVRTC options
```

### 4. PTX Compatibility
```bash
# Test if SM100 PTX can JIT to SM120e
CUDA_FORCE_PTX_JIT=1 python -c "import deep_gemm"
# Expected: may work for basic kernels, will fail for tcgen05.fence
```

## JIT Cache

DeepGEMM caches compiled kernels. When testing SM120e changes:
```bash
# Clear JIT cache
rm -rf ~/.cache/deep_gemm/
# Or set custom cache dir
export DEEPGEMM_CACHE_DIR=/tmp/deepgemm_sm120e
```
