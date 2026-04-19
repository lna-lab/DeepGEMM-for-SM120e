#!/bin/bash
# Bootstrap DeepGEMM-for-SM120e development environment
# Clones upstream DeepGEMM as a reference and sets up workspace

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== DeepGEMM-for-SM120e Bootstrap ==="

# 1. Clone upstream DeepGEMM as reference (not submodule — we diverge)
UPSTREAM_DIR="${PROJECT_DIR}/upstream_ref"
if [ ! -d "$UPSTREAM_DIR" ]; then
    echo "[1/3] Cloning upstream DeepGEMM as reference..."
    git clone --depth 1 https://github.com/deepseek-ai/DeepGEMM.git "$UPSTREAM_DIR"
else
    echo "[1/3] Upstream reference already present"
fi

# 2. Run environment check
echo "[2/3] Checking environment..."
bash "${SCRIPT_DIR}/test_sm120e_env.sh"

# 3. Create venv for development (optional)
VENV_DIR="${PROJECT_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[3/3] Creating development venv..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install torch numpy pytest
    echo "Venv created at $VENV_DIR"
else
    echo "[3/3] Venv already exists"
fi

echo ""
echo "=== Bootstrap complete ==="
echo "  Upstream ref: $UPSTREAM_DIR"
echo "  Dev venv:     $VENV_DIR"
echo ""
echo "Next steps:"
echo "  1. Read docs/roadmap.md"
echo "  2. Read docs/compatibility.md"
echo "  3. Start with architecture detection in upstream_ref/"
