"""Correctness validation: DeepGEMM-SM120e vs PyTorch reference.

Every ported kernel must pass these tests before benchmarking.
"""

import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestCorrectnessVsTorch:
    """Correctness tests — populated as kernels are ported."""

    @pytest.mark.skip(reason="SM120e kernel not yet implemented")
    def test_dense_gemm_matches_torch(self):
        """DeepGEMM dense GEMM output matches torch.mm within tolerance."""
        pass

    @pytest.mark.skip(reason="SM120e kernel not yet implemented")
    def test_grouped_gemm_matches_batched_torch(self):
        """DeepGEMM grouped GEMM matches batched torch.bmm."""
        pass

    @pytest.mark.skip(reason="SM120e kernel not yet implemented")
    def test_scaling_factors_correct(self):
        """Verify FP8 scaling factor application matches reference."""
        pass
