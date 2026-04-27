#!/usr/bin/env bash
# patch_deepgemm_sm120e.sh — Patch /opt/deepgemm for SM120e, build, and prepare
#
# Run INSIDE the Docker container:
#   docker compose run --rm dev bash
#   bash /workspace/DeepGEMM-for-SM120e/scripts/patch_deepgemm_sm120e.sh
#
# What it does:
#   1. Patches arch detection to treat SM120 as SM100 (same Blackwell ISA minus fence)
#   2. Patches JIT compiler to target SM120a
#   3. Copies our SM120e kernel header
#   4. Builds DeepGEMM with patches applied

set -euo pipefail

UPSTREAM=/opt/deepgemm
WORKSPACE=/workspace/DeepGEMM-for-SM120e

echo "=== Patching DeepGEMM for SM120e ==="

# 1. Patch arch detection: SM120 (major=12) → treated as SM100 (major=10)
echo "[1/5] Patching arch detection..."
# get_arch_major(): return 10 for SM120 so all arch_major==10 checks pass
sed -i 's|int get_arch_major() {|int get_arch_major() {\n        // SM120e patch: treat SM120 as SM100 for API compatibility\n        if (get_arch_pair().first == 12) return 10;|' \
    "$UPSTREAM/csrc/jit/device_runtime.hpp"

# get_arch(): SM120 compiles as sm_100f (Blackwell family).
# tcgen05 instructions are part of the Blackwell ISA family.
# sm_120a rejects tcgen05, but sm_100f accepts it and should run on SM120 hardware.
sed -i '/if (major == 10 and minor != 1)/i\        // SM120e patch: compile as sm_100f (Blackwell family)\n        if (major == 12) {\n            if (number_only) return "100";\n            return support_arch_family ? "100f" : "100a";\n        }' \
    "$UPSTREAM/csrc/jit/device_runtime.hpp"

echo "  -> Patched device_runtime.hpp"

# 2. Patch get_default_recipe(): add SM120 support (same as SM100)
echo "[2/5] Patching recipe detection..."
sed -i 's|} else if (arch_major == 10) {|} else if (arch_major == 10 or arch_major == 12) {|' \
    "$UPSTREAM/csrc/utils/layout.hpp"
echo "  -> Patched layout.hpp"

# 3. Copy our SM120e kernel header
echo "[3/5] Installing SM120e kernel header..."
cp "$WORKSPACE/src/include/deep_gemm/impls/sm120e_fp8_gemm_1d1d.cuh" \
   "$UPSTREAM/deep_gemm/include/deep_gemm/impls/"
echo "  -> Installed sm120e_fp8_gemm_1d1d.cuh"

# 4. Build DeepGEMM
echo "[4/5] Building DeepGEMM..."
cd "$UPSTREAM"
pip install -e . --no-build-isolation --break-system-packages 2>&1 | tail -3
echo "  -> Build complete"

# 5. Verify import
echo "[5/5] Verifying import..."
python -c "import deep_gemm; print(f'  -> deep_gemm {deep_gemm.__version__} loaded')"

echo ""
echo "=== Patch complete ==="
echo "Run test:"
echo "  python /workspace/DeepGEMM-for-SM120e/tests/test_fp8_gemm_sm120e.py"
