"""SM120e FP8 GEMM test via CUTLASS PyTorch extension.

Usage (inside Docker):
  cd /workspace/DeepGEMM-for-SM120e/src
  pip install -e . --break-system-packages
  python /workspace/DeepGEMM-for-SM120e/tests/test_sm120e_cutlass.py
"""

import sys
import torch


def check_env():
    cc = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (SM{cc[0]*10 + cc[1]})")
    return cc[0] >= 12


def test_fp8_gemm():
    import sm120e_gemm

    print("\n=== SM120e CUTLASS FP8 GEMM Test ===")

    test_shapes = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]

    all_passed = True
    for M, N, K in test_shapes:
        try:
            # Create BF16 source data
            a_bf16 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
            b_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)

            # Reference: BF16 matmul
            ref = (a_bf16.float() @ b_bf16.float().T).to(torch.bfloat16)

            # Quantize to FP8
            a_fp8 = a_bf16.to(torch.float8_e4m3fn)
            b_fp8 = b_bf16.to(torch.float8_e4m3fn)

            # Get SF shapes from extension
            sfa_shape, sfb_shape = sm120e_gemm.get_sf_shapes(M, N, K)

            # Create trivial scale factors (all 1.0)
            sf_a = torch.ones(sfa_shape, device="cuda", dtype=torch.float32)
            sf_b = torch.ones(sfb_shape, device="cuda", dtype=torch.float32)

            # Run CUTLASS GEMM
            D = sm120e_gemm.fp8_gemm(a_fp8, sf_a, b_fp8, sf_b)

            # Compare
            diff = (D.float() - ref.float()).abs().mean() / ref.float().abs().mean()
            # FP8 quantization introduces ~1% error with trivial SF
            status = "PASS" if diff < 0.05 else "FAIL"
            if diff >= 0.05:
                all_passed = False
            print(f"  {status}: M={M:4d}, N={N:4d}, K={K:4d} | diff={diff:.6f}")

        except Exception as e:
            print(f"  ERROR: M={M:4d}, N={N:4d}, K={K:4d} | {type(e).__name__}: {e}")
            all_passed = False

    return all_passed


def bench_fp8_gemm():
    import sm120e_gemm

    print("\n=== SM120e CUTLASS FP8 GEMM Benchmark ===")

    shapes = [
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (4096, 4096, 8192),
    ]

    for M, N, K in shapes:
        a_fp8 = torch.randn(M, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        b_fp8 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        sfa_shape, sfb_shape = sm120e_gemm.get_sf_shapes(M, N, K)
        sf_a = torch.ones(sfa_shape, device="cuda", dtype=torch.float32)
        sf_b = torch.ones(sfb_shape, device="cuda", dtype=torch.float32)

        # Warmup
        for _ in range(5):
            D = sm120e_gemm.fp8_gemm(a_fp8, sf_a, b_fp8, sf_b)
        torch.cuda.synchronize()

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        n_iter = 50
        start.record()
        for _ in range(n_iter):
            D = sm120e_gemm.fp8_gemm(a_fp8, sf_a, b_fp8, sf_b)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / n_iter
        tflops = 2 * M * N * K / elapsed_ms / 1e9
        print(f"  {M:4d}x{N:4d}x{K:5d} | {elapsed_ms:.4f} ms | {tflops:.0f} TFLOPS")


if __name__ == "__main__":
    if not check_env():
        print("Wrong GPU")
        sys.exit(1)

    ok = test_fp8_gemm()
    if ok:
        bench_fp8_gemm()

    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    sys.exit(0 if ok else 1)
