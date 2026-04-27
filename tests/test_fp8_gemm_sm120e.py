"""SM120e FP8 GEMM correctness test.

Uses upstream DeepGEMM's own test generators for correct SF layout.

Usage (inside Docker):
  bash /workspace/DeepGEMM-for-SM120e/scripts/patch_deepgemm_sm120e.sh
  python /workspace/DeepGEMM-for-SM120e/tests/test_fp8_gemm_sm120e.py
"""

import sys
import os
import torch

# Add upstream tests to path for generators
sys.path.insert(0, "/opt/deepgemm/tests")


def check_env():
    """Verify SM120e environment."""
    assert torch.cuda.is_available(), "CUDA not available"
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (SM{cc[0]*10 + cc[1]})")
    print(f"Compute capability: {cc[0]}.{cc[1]}")
    props = torch.cuda.get_device_properties(0)
    print(f"Shared memory/block: {props.shared_memory_per_block:,} bytes")
    return cc[0] >= 12


def calc_diff(a, b):
    """Relative difference (same as DeepGEMM)."""
    a, b = a.to(torch.float64), b.to(torch.float64)
    return (a - b).abs().mean() / b.abs().mean()


def test_deepgemm_fp8_nt():
    """Test fp8_gemm_nt using upstream generators for correct SF layout."""
    import deep_gemm
    from deep_gemm.testing import calc_diff as dg_calc_diff
    from generators import (
        KernelType, MajorTypeAB, QuantConfig,
        generate_normal, get_ue8m0_usage, reset_seed
    )

    print("\n=== Testing DeepGEMM fp8_gemm_nt (SM120e) ===")

    # Test shapes (M, N, K) — multiples of BLOCK tiles
    test_shapes = [
        (64,   64,   128),    # 1 tile
        (128,  128,  128),    # 2x2 tiles
        (128,  128,  256),    # Multi K
        (256,  256,  512),    # Larger
        (64,   256,  1024),   # Skinny M
        (512,  512,  1024),   # Medium
    ]

    kernel_type = KernelType.Kernel1D1D
    major_a = MajorTypeAB.KMajor
    major_b = MajorTypeAB.KMajor
    accumulate = False
    out_dtype = torch.bfloat16
    quant_config = QuantConfig()  # Default: gran_k_a=128, gran_k_b=128
    use_ue8m0 = get_ue8m0_usage(kernel_type)

    all_passed = True
    for M, N, K in test_shapes:
        try:
            reset_seed(42)
            a, b, c, d, ref_d = generate_normal(
                M, N, K, major_a, major_b,
                accumulate, out_dtype, kernel_type,
                use_ue8m0=use_ue8m0, quant_config=quant_config
            )

            # Call DeepGEMM
            deep_gemm.fp8_gemm_nt(a, b, d)

            diff = dg_calc_diff(d, ref_d)
            max_diff = quant_config.max_diff()  # 0.001 for FP8xFP8
            status = "PASS" if diff < max_diff else "FAIL"
            if diff >= max_diff:
                all_passed = False
            print(f"  {status}: M={M:4d}, N={N:4d}, K={K:4d} | diff={diff:.6f} (max={max_diff})")

        except Exception as e:
            print(f"  ERROR: M={M:4d}, N={N:4d}, K={K:4d} | {type(e).__name__}: {e}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print("=" * 60)
    print("SM120e FP8 GEMM Correctness Test")
    print("=" * 60)

    if not check_env():
        print("\nSkipping (wrong GPU arch)")
        sys.exit(1)

    ok = test_deepgemm_fp8_nt()

    print("\n" + "=" * 60)
    if ok:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    sys.exit(0 if ok else 1)
