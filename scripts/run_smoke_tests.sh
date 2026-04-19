#!/bin/bash
# Run smoke tests for SM120e kernels
# Execute after a kernel has been ported

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== SM120e Smoke Tests ==="

cd "$PROJECT_DIR"

# Activate venv if exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Run tests
python3 -m pytest tests/ -v --tb=short 2>&1

echo "=== Smoke tests complete ==="
