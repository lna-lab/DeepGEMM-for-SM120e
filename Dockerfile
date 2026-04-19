# ============================================================================
# DeepGEMM-for-SM120e Development Container
# ============================================================================
# Isolated build environment with CUDA 13.0, CUTLASS 4.0, C++20
# For kernel development and testing on SM120e (RTX PRO 6000)
# ============================================================================

FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

LABEL maintainer="Lna-Lab"
LABEL description="DeepGEMM SM120e kernel development environment"

ENV DEBIAN_FRONTEND=noninteractive

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        git curl build-essential cmake ninja-build \
        && rm -rf /var/lib/apt/lists/* \
        && ln -sf /usr/bin/python3 /usr/bin/python

# ── Python environment ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir --break-system-packages \
        torch==2.11.0 \
        numpy \
        pytest \
        packaging

# ── Clone DeepGEMM with CUTLASS submodule ───────────────────────────────────
RUN git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git /opt/deepgemm

# ── Build DeepGEMM (will fail on SM120e — that's what we're fixing) ──────────
WORKDIR /opt/deepgemm

# Link CUTLASS
RUN ln -sf /opt/deepgemm/third-party/cutlass/include/cutlass deep_gemm/include && \
    ln -sf /opt/deepgemm/third-party/cutlass/include/cute deep_gemm/include

# ── Working directory for SM120e port ────────────────────────────────────────
RUN mkdir -p /workspace
WORKDIR /workspace

# ── Environment ──────────────────────────────────────────────────────────────
ENV CUDA_HOME=/usr/local/cuda
ENV DEEPGEMM_ROOT=/opt/deepgemm
ENV PYTHONPATH=/opt/deepgemm:$PYTHONPATH

CMD ["/bin/bash"]
