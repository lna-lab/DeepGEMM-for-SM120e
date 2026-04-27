// sm120e_gemm.cu — PyTorch C++ extension for SM120 FP8 GEMM via CUTLASS
//
// Wraps CUTLASS example 87a pattern into a torch-callable function.
// Input:  FP8 E4M3 tensors A (M,K), B (N,K) + FP32 blockwise scale factors
// Output: BF16 tensor D (M,N)

#include <torch/extension.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"

using namespace cute;

// ============================================================================
// CUTLASS SM120 FP8 Blockwise-Scaled GEMM type definition
// ============================================================================

using ElementA            = cutlass::float_e4m3_t;
using LayoutA             = cutlass::layout::RowMajor;      // A is (M, K) row-major
constexpr int AlignmentA  = 16;                             // 128 bits / 8 bits

using ElementB            = cutlass::float_e4m3_t;
using LayoutB             = cutlass::layout::ColumnMajor;   // B is (K, N) = (N, K).T
constexpr int AlignmentB  = 16;

using ElementC            = cutlass::bfloat16_t;
using LayoutC             = cutlass::layout::RowMajor;
constexpr int AlignmentC  = 8;                              // 128 bits / 16 bits

using ElementD            = cutlass::bfloat16_t;
using LayoutD             = cutlass::layout::RowMajor;
constexpr int AlignmentD  = 8;

using ElementAccumulator  = float;
using ElementCompute      = float;

using MmaTileShape_MNK    = Shape<_128, _128, _128>;
using ClusterShape_MNK    = Shape<_1, _1, _1>;

// Scale factor layout deduction
using ScaleConfig = decltype(cutlass::detail::sm120_trivial_blockwise_scale_config(MmaTileShape_MNK{}));
using LayoutSFA   = decltype(ScaleConfig::deduce_layoutSFA());
using LayoutSFB   = decltype(ScaleConfig::deduce_layoutSFB());

// Epilogue
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

// Mainloop — note the cute::tuple<Layout, LayoutSF> for blockwise scaling
using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp,
    ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
    ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
    ElementAccumulator,
    MmaTileShape_MNK, ClusterShape_MNK,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

// ============================================================================
// PyTorch-callable function
// ============================================================================

torch::Tensor sm120e_fp8_gemm(
    torch::Tensor A,        // (M, K) float8_e4m3fn
    torch::Tensor sf_A,     // Scale factors for A (FP32, blockwise)
    torch::Tensor B,        // (N, K) float8_e4m3fn — note: N x K, not K x N
    torch::Tensor sf_B,     // Scale factors for B (FP32, blockwise)
    float alpha,
    float beta
) {
    TORCH_CHECK(A.is_cuda(), "A must be on CUDA");
    TORCH_CHECK(B.is_cuda(), "B must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat8_e4m3fn, "A must be float8_e4m3fn");
    TORCH_CHECK(B.dtype() == torch::kFloat8_e4m3fn, "B must be float8_e4m3fn");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);
    TORCH_CHECK(B.size(1) == K, "B.size(1) must equal A.size(1)");

    // Ensure contiguity
    A = A.contiguous();
    B = B.contiguous();
    sf_A = sf_A.contiguous();
    sf_B = sf_B.contiguous();

    // Allocate output
    auto D = torch::empty({M, N}, torch::TensorOptions().device(A.device()).dtype(torch::kBFloat16));

    // Build strides
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Build scale factor layouts
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    // Build GEMM arguments
    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ElementA*>(A.data_ptr()),
            stride_A,
            reinterpret_cast<ElementB*>(B.data_ptr()),
            stride_B,
            reinterpret_cast<ElementAccumulator*>(sf_A.data_ptr()),
            layout_SFA,
            reinterpret_cast<ElementAccumulator*>(sf_B.data_ptr()),
            layout_SFB
        },
        {
            {},  // epilogue.thread (fusion args)
            nullptr,  // C pointer (nullptr for beta=0)
            stride_C,
            reinterpret_cast<ElementD*>(D.data_ptr()),
            stride_D
        }
    };
    arguments.epilogue.thread.alpha = alpha;
    arguments.epilogue.thread.beta = beta;

    // Launch
    Gemm gemm;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    auto workspace = torch::empty({static_cast<long>(workspace_size)},
                                   torch::TensorOptions().device(A.device()).dtype(torch::kUInt8));

    auto status = gemm.can_implement(arguments);
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS cannot implement this GEMM: ", cutlass::cutlassGetStatusString(status));

    status = gemm.initialize(arguments, workspace.data_ptr());
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS init failed: ", cutlass::cutlassGetStatusString(status));

    status = gemm.run();
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS run failed: ", cutlass::cutlassGetStatusString(status));

    return D;
}

// Convenience: compute SF layout sizes for a given problem shape
std::tuple<std::vector<int64_t>, std::vector<int64_t>> sm120e_get_sf_shapes(int M, int N, int K) {
    LayoutSFA layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    LayoutSFB layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    auto sfa_size = static_cast<int64_t>(cute::size(cute::filter_zeros(layout_SFA)));
    auto sfb_size = static_cast<int64_t>(cute::size(cute::filter_zeros(layout_SFB)));

    return {{sfa_size}, {sfb_size}};
}

// ============================================================================
// Python bindings
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fp8_gemm", &sm120e_fp8_gemm,
          "SM120e FP8 E4M3 GEMM with blockwise scaling (CUTLASS)",
          py::arg("A"), py::arg("sf_A"),
          py::arg("B"), py::arg("sf_B"),
          py::arg("alpha") = 1.0f,
          py::arg("beta") = 0.0f);

    m.def("get_sf_shapes", &sm120e_get_sf_shapes,
          "Get scale factor tensor sizes for given M, N, K",
          py::arg("M"), py::arg("N"), py::arg("K"));
}
