"""Dense GEMM correctness tests for SM120e.

Tests fp8_gemm_nt against PyTorch reference implementation.
"""

import pytest
import torch


def reference_fp8_gemm(A_fp8, B_fp8, scale_a, scale_b):
    """Reference implementation: dequantize → matmul → result."""
    A_f32 = A_fp8.to(torch.float32) * scale_a
    B_f32 = B_fp8.to(torch.float32) * scale_b
    return A_f32 @ B_f32.T


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)
class TestDenseGEMM:
    """Dense GEMM tests — will be populated as kernels are ported."""

    def test_cuda_available(self):
        assert torch.cuda.is_available()
        cc = torch.cuda.get_device_capability()
        # SM120 = compute capability 12.0
        assert cc[0] >= 12, f"Expected SM120+, got sm_{cc[0]*10+cc[1]}"

    def test_fp8_dtype_available(self):
        assert hasattr(torch, "float8_e4m3fn"), "FP8 E4M3 not available"
        x = torch.randn(4, 4, device="cuda").to(torch.float8_e4m3fn)
        assert x.dtype == torch.float8_e4m3fn

    def test_reference_gemm(self):
        """Verify reference implementation works."""
        M, N, K = 64, 64, 32
        A = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
        B = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
        result = reference_fp8_gemm(A, B, scale_a=1.0, scale_b=1.0)
        assert result.shape == (M, N)

    @pytest.mark.skip(reason="SM120e kernel not yet implemented")
    def test_fp8_gemm_nt_small(self):
        """Small dense GEMM — first kernel to port."""
        pass

    @pytest.mark.skip(reason="SM120e kernel not yet implemented")
    def test_fp8_gemm_nt_shapes(self):
        """Test various shapes relevant to LLM inference."""
        pass
