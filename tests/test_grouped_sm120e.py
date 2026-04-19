"""Grouped GEMM tests for SM120e (MoE workloads).

Tests grouped GEMM kernels against batched PyTorch reference.
"""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestGroupedGEMM:
    """Grouped GEMM tests — will be populated as kernels are ported."""

    @pytest.mark.skip(reason="SM120e kernel not yet implemented")
    def test_grouped_gemm_masked(self):
        """MoE decoding: variable expert loads per token."""
        pass

    @pytest.mark.skip(reason="SM120e kernel not yet implemented")
    def test_grouped_gemm_contiguous(self):
        """MoE prefill: contiguous expert dispatch."""
        pass

    @pytest.mark.skip(reason="SM120e kernel not yet implemented")
    def test_qwen36_expert_shapes(self):
        """Qwen3.6 MoE: 256 experts, intermediate_dim=512, hidden=2048."""
        pass
