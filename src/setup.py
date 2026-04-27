"""Build SM120e FP8 GEMM PyTorch extension."""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUTLASS_ROOT = os.environ.get("CUTLASS_ROOT", "/opt/deepgemm/third-party/cutlass")

setup(
    name="sm120e_gemm",
    ext_modules=[
        CUDAExtension(
            name="sm120e_gemm",
            sources=["csrc/sm120e_gemm.cu"],
            include_dirs=[
                f"{CUTLASS_ROOT}/include",
                f"{CUTLASS_ROOT}/tools/util/include",
            ],
            extra_compile_args={
                "cxx": ["-std=c++20", "-O3"],
                "nvcc": [
                    "-std=c++20",
                    "-arch=sm_120a",
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-DCUTLASS_ARCH_MMA_SM120_SUPPORTED=1",
                    "--diag-suppress=2908",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
