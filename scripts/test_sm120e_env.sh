#!/bin/bash
# Test SM120e development environment
# Run this first to verify prerequisites

set -e

echo "=== SM120e Environment Check ==="
echo ""

# 1. GPU
echo "[GPU]"
nvidia-smi --query-gpu=index,name,compute_cap,driver_version --format=csv,noheader | head -1
echo ""

# 2. CUDA
echo "[CUDA]"
nvcc --version 2>/dev/null | grep "release" || echo "nvcc not found"
echo ""

# 3. Python + PyTorch
echo "[Python/PyTorch]"
python3 -c "
import torch
print(f'Python: {__import__(\"sys\").version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    cc = torch.cuda.get_device_capability(0)
    print(f'Compute capability: {cc[0]}.{cc[1]}')
    print(f'SM class: sm_{cc[0]*10 + cc[1]}')
    props = torch.cuda.get_device_properties(0)
    print(f'Shared memory / block: {props.max_shared_memory_per_block} bytes')
    print(f'Total memory: {props.total_mem / 1e9:.1f} GB')
" 2>&1

echo ""

# 4. CUTLASS
echo "[CUTLASS]"
if [ -d "/usr/local/cutlass" ] || python3 -c "import cutlass" 2>/dev/null; then
    echo "CUTLASS: found"
else
    echo "CUTLASS: not found (optional)"
fi
echo ""

# 5. FP8 support
echo "[FP8 Tensor Core]"
python3 -c "
import torch
if hasattr(torch, 'float8_e4m3fn'):
    a = torch.randn(16, 16, device='cuda').to(torch.float8_e4m3fn)
    print(f'FP8 E4M3: supported (tensor shape {a.shape})')
else:
    print('FP8: not supported in this PyTorch version')
" 2>&1

echo ""
echo "=== Environment check complete ==="
