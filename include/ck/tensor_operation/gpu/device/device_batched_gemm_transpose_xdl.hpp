#ifndef DEVICE_BATCHED_GEMM_TRANSPOSE_XDL_HPP
#define DEVICE_BATCHED_GEMM_TRANSPOSE_XDL_HPP

#include <iostream>
#include <sstream>
#include "device.hpp"
#include "device_base.hpp"
#include "device_batched_gemm.hpp"
#include "common_header.hpp"
#include "tensor_layout.hpp"
#include "tensor_descriptor.hpp"
#include "tensor_descriptor_helper.hpp"
#include "gridwise_gemm_xdlops_v2r3.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

/*
 * \brief Wrapper function of GridwiseGemm::Run to realize BatchedGEMM.
 *
 * \tparam ComputePtrOffsetOfBatch Class that computes the base pointer offsets of A, B, C matrix
 * given the batch. For example, ComputePtrOffsetOfStridedBatch() computes the offsets of evenly
 * strided batched, but we can easily extend to other layouts. The returned offset can be either \p
 * index_t or \p long_index_t. If it returns \p long_index_t, we are not subject to the 2GB
 * limitations.
 *
 * \tparam Block2CTileMap Block2CTileMap::CalculateBottomIndex() takes in id of a workgroup and
 * returns the 2D index of the tile that it computes. \see
 * GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3::Run().
 *
 * \note Using \p ComputePtrOffsetOfBatch gives us the flexibility that 2 workgroups can compute 2
 * tiles from different matrices. Keep in mind that these 2 matrices can share the same grid
 * descriptor (like in BatchedGEMM), or use their own grid descriptors (in GroupedGemm). \link
 * device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp kernel_gemm_xdlops_v2r3_for_conv3d \endlink for \link
 * DeviceConv3d \endlink uses the same concept, but currently does NOT encapsulate the computing of
 * pointer offset into \p ComputePtrOffsetOfStridedBatch.
 *
 * \note \p Block2CTileMap allows customized mapping between a workgroup and the C-tile it computes.
 * Together with \p ComputePtrOffsetOfBatch, we can reuse GridwiseGemm (and GridwiseGemm fusion ) to
 * realize BatchedGemmTranspose and GroupedGemm (and the corresponding GEMM fusion).
 *
 */
template <typename GridwiseGemm,
          typename FloatAB,
          typename FloatC,
          typename AGridDesc_K0_M_K1,
          typename BGridDesc_K0_N_K1,
          typename CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          typename ComputePtrOffsetOfBatch,
          typename Block2CTileMap,
          bool HasMainKBlockLoop>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_batched_gemm_transpose_xdlops_v2r3(
            const FloatAB* __restrict__ p_a_grid,
            const FloatAB* __restrict__ p_b_grid,
            FloatC* __restrict__ p_c_grid,
            const index_t batch_count,
            const AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1,
            const BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1,
            const CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
            const AElementwiseOperation a_element_op,
            const BElementwiseOperation b_element_op,
            const CElementwiseOperation c_element_op,
            const ComputePtrOffsetOfBatch compute_ptr_offset_of_batch,
            const Block2CTileMap block_2_ctile_map)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__))
    const index_t num_blocks_per_batch =
        __builtin_amdgcn_readfirstlane(get_grid_size() / batch_count);
    const index_t g_idx = __builtin_amdgcn_readfirstlane(get_block_1d_id() / num_blocks_per_batch);

    const long_index_t a_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetAPtrOffset(g_idx)));
    const long_index_t b_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetBPtrOffset(g_idx)));
    const long_index_t c_batch_offset = __builtin_amdgcn_readfirstlane(
        static_cast<long_index_t>(compute_ptr_offset_of_batch.GetCPtrOffset(g_idx)));

    __shared__ char p_shared[GridwiseGemm::GetSharedMemoryNumberOfByte()];

    GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid + a_batch_offset,
                                                  p_b_grid + b_batch_offset,
                                                  p_c_grid + c_batch_offset,
                                                  p_shared,
                                                  a_grid_desc_k0_m_k1,
                                                  b_grid_desc_k0_n_k1,
                                                  c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2,
                                                  a_element_op,
                                                  b_element_op,
                                                  c_element_op,
                                                  block_2_ctile_map);
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = batch_count;
    ignore = a_grid_desc_k0_m_k1;
    ignore = b_grid_desc_k0_n_k1;
    ignore = c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
    ignore = compute_ptr_offset_of_batch;
    ignore = block_2_ctile_map;
#endif // end of if (defined(__gfx908__) || defined(__gfx90a__))
}

template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          ck::index_t BlockSize,
          ck::index_t MPerBlock,
          ck::index_t NPerBlock,
          ck::index_t K0PerBlock,
          ck::index_t K1,
          ck::index_t MPerXDL,
          ck::index_t NPerXDL,
          ck::index_t MXdlPerWave,
          ck::index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          ck::index_t ABlockTransferSrcVectorDim,
          ck::index_t ABlockTransferSrcScalarPerVector,
          ck::index_t ABlockTransferDstScalarPerVector_K1,
          bool ABlockLdsAddExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          ck::index_t BBlockTransferSrcVectorDim,
          ck::index_t BBlockTransferSrcScalarPerVector,
          ck::index_t BBlockTransferDstScalarPerVector_K1,
          bool BBlockLdsAddExtraN,
          ck::index_t CThreadTransferSrcDstVectorDim,
          ck::index_t CThreadTransferDstScalarPerVector>
struct DeviceBatchedGemmTransposeXdl : public DeviceBatchedGemmTranspose<AElementwiseOperation,
                                                                         BElementwiseOperation,
                                                                         CElementwiseOperation>
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};

    static constexpr auto K1Number = Number<K1>{};

    static auto MakeAGridDescriptor_K0_M_K1(index_t M, index_t K, index_t StrideA)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;

        const auto a_grid_desc_k0_mp_k1 =
            transform_tensor_descriptor(a_grid_desc_m_k,
                                        make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                                                   make_right_pad_transform(M, PadM)),
                                        make_tuple(Sequence<1>{}, Sequence<0>{}),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return a_grid_desc_k0_mp_k1;
    }

    static auto MakeBGridDescriptor_K0_N_K1(index_t K, index_t N, index_t StrideB)
    {
        assert(K % K1 == 0);

        const index_t K0 = K / K1;

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

        const auto b_grid_desc_k0_np_k1 =
            transform_tensor_descriptor(b_grid_desc_k_n,
                                        make_tuple(make_unmerge_transform(make_tuple(K0, K1Number)),
                                                   make_right_pad_transform(N, PadN)),
                                        make_tuple(Sequence<0>{}, Sequence<1>{}),
                                        make_tuple(Sequence<0, 2>{}, Sequence<1>{}));

        return b_grid_desc_k0_np_k1;
    }

    static auto MakeCGridDescriptor_M_N(index_t M0,
                                        index_t M1,
                                        index_t N0,
                                        index_t N1,
                                        index_t StrideM0,
                                        index_t StrideM1,
                                        index_t StrideN0,
                                        index_t StrideN1)
    {
        const auto c_grid_desc_m_n = [&]() {
            const auto c_grid_desc_m0_m1_n0_n1 = make_naive_tensor_descriptor(
                make_tuple(M0, M1, N0, N1), make_tuple(StrideM0, StrideM1, StrideN0, StrideN1));

            return transform_tensor_descriptor(c_grid_desc_m0_m1_n0_n1,
                                               make_tuple(make_merge_transform(make_tuple(M0, M1)),
                                                          make_merge_transform(make_tuple(N0, N1))),
                                               make_tuple(Sequence<0, 1>{}, Sequence<2, 3>{}),
                                               make_tuple(Sequence<0>{}, Sequence<1>{}));
        }();

        const index_t M = M0 * M1;
        const index_t N = N0 * N1;

        const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
        const auto PadN = (NPerBlock - N % NPerBlock) % NPerBlock;

        const auto c_grid_desc_mp_np = transform_tensor_descriptor(
            c_grid_desc_m_n,
            make_tuple(make_right_pad_transform(M, PadM), make_right_pad_transform(N, PadN)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return c_grid_desc_mp_np;
    }

    using AGridDesc_K0_M_K1 = decltype(MakeAGridDescriptor_K0_M_K1(1, 1, 1));
    using BGridDesc_K0_N_K1 = decltype(MakeBGridDescriptor_K0_N_K1(1, 1, 1));
    using CGridDesc_M_N     = decltype(MakeCGridDescriptor_M_N(1, 1, 1, 1, 1, 1, 1, 1));

    struct ComputePtrOffsetOfStridedBatch
    {
        ComputePtrOffsetOfStridedBatch(index_t BatchStrideA,
                                       index_t BatchStrideB,
                                       index_t BatchStrideC)
            : BatchStrideA_(BatchStrideA), BatchStrideB_(BatchStrideB), BatchStrideC_(BatchStrideC)
        {
        }

        __host__ __device__ constexpr long_index_t GetAPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideA_);
        }

        __host__ __device__ constexpr long_index_t GetBPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideB_);
        }

        __host__ __device__ constexpr long_index_t GetCPtrOffset(index_t g_idx) const
        {
            return g_idx * static_cast<long_index_t>(BatchStrideC_);
        }

        private:
        index_t BatchStrideA_;
        index_t BatchStrideB_;
        index_t BatchStrideC_;
    };

    // GridwiseGemm
    using GridwiseGemm =
        GridwiseGemm_k0mk1_k0nk1_mn_xdlops_v2r3<BlockSize,
                                                ADataType, // TODO: distinguish A/B datatype
                                                AccDataType,
                                                CDataType,
                                                InMemoryDataOperationEnum::Set,
                                                AGridDesc_K0_M_K1,
                                                BGridDesc_K0_N_K1,
                                                CGridDesc_M_N,
                                                AElementwiseOperation,
                                                BElementwiseOperation,
                                                CElementwiseOperation,
                                                MPerBlock,
                                                NPerBlock,
                                                K0PerBlock,
                                                MPerXDL,
                                                NPerXDL,
                                                K1,
                                                MXdlPerWave,
                                                NXdlPerWave,
                                                ABlockTransferThreadClusterLengths_K0_M_K1,
                                                ABlockTransferThreadClusterArrangeOrder,
                                                ABlockTransferSrcAccessOrder,
                                                ABlockTransferSrcVectorDim,
                                                ABlockTransferSrcScalarPerVector,
                                                ABlockTransferDstScalarPerVector_K1,
                                                false, // AThreadTransferSrcResetCoordinateAfterRun,
                                                ABlockLdsAddExtraM,
                                                BBlockTransferThreadClusterLengths_K0_N_K1,
                                                BBlockTransferThreadClusterArrangeOrder,
                                                BBlockTransferSrcAccessOrder,
                                                BBlockTransferSrcVectorDim,
                                                BBlockTransferSrcScalarPerVector,
                                                BBlockTransferDstScalarPerVector_K1,
                                                false, // BThreadTransferSrcResetCoordinateAfterRun,
                                                BBlockLdsAddExtraN,
                                                Sequence<2, 3, 0, 1, 7, 5, 4, 6>,
                                                CThreadTransferSrcDstVectorDim,
                                                CThreadTransferDstScalarPerVector>;

    using CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 =
        decltype(GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(CGridDesc_M_N{}));
    using Block2CTileMap = typename GridwiseGemm::DefaultBlock2CTileMap;

    // Argument
    struct Argument : public BaseArgument
    {
        Argument(const ADataType* p_a_grid,
                 const BDataType* p_b_grid,
                 CDataType* p_c_grid,
                 GemmTransposeDesc gemm_transpose_desc,
                 index_t M01,
                 index_t N01,
                 AElementwiseOperation a_element_op,
                 BElementwiseOperation b_element_op,
                 CElementwiseOperation c_element_op,
                 index_t BatchCount)
            : p_a_grid_{p_a_grid},
              p_b_grid_{p_b_grid},
              p_c_grid_{p_c_grid},
              BatchCount_(BatchCount),
              a_grid_desc_k0_m_k1_{DeviceBatchedGemmTransposeXdl::MakeAGridDescriptor_K0_M_K1(
                  gemm_transpose_desc.M_, gemm_transpose_desc.K_, gemm_transpose_desc.stride_A_)},
              b_grid_desc_k0_n_k1_{DeviceBatchedGemmTransposeXdl::MakeBGridDescriptor_K0_N_K1(
                  gemm_transpose_desc.K_, gemm_transpose_desc.N_, gemm_transpose_desc.stride_B_)},
              c_grid_desc_m_n_{DeviceBatchedGemmTransposeXdl::MakeCGridDescriptor_M_N(
                  gemm_transpose_desc.M0_,
                  gemm_transpose_desc.M1_,
                  gemm_transpose_desc.N0_,
                  gemm_transpose_desc.N1_,
                  gemm_transpose_desc.stride_M0_,
                  gemm_transpose_desc.stride_M1_,
                  gemm_transpose_desc.stride_N0_,
                  gemm_transpose_desc.stride_N1_)},
              c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_{},
              compute_ptr_offset_of_batch_{
                  type_convert<index_t>(a_grid_desc_k0_m_k1_.GetElementSpaceSize()),
                  type_convert<index_t>(b_grid_desc_k0_n_k1_.GetElementSpaceSize()),
                  type_convert<index_t>(c_grid_desc_m_n_.GetElementSpaceSize())},
              block_2_ctile_map_{
                  GridwiseGemm::MakeDefaultBlock2CTileMap(c_grid_desc_m_n_, M01, N01)},
              M01_{M01},
              N01_{N01},
              a_element_op_{a_element_op},
              b_element_op_{b_element_op},
              c_element_op_{c_element_op}
        {
            if(GridwiseGemm::CheckValidity(a_grid_desc_k0_m_k1_,
                                           b_grid_desc_k0_n_k1_,
                                           c_grid_desc_m_n_,
                                           block_2_ctile_map_))
            {
                c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_ =
                    GridwiseGemm::MakeCGridDescriptor_M0_N0_M1_N1_M2_M3_M4_N2(c_grid_desc_m_n_);
            }
        }

        //  private:
        const ADataType* p_a_grid_;
        const BDataType* p_b_grid_;
        CDataType* p_c_grid_;
        index_t BatchCount_;
        AGridDesc_K0_M_K1 a_grid_desc_k0_m_k1_;
        BGridDesc_K0_N_K1 b_grid_desc_k0_n_k1_;
        CGridDesc_M_N c_grid_desc_m_n_;
        CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2 c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_;
        ComputePtrOffsetOfStridedBatch compute_ptr_offset_of_batch_;
        Block2CTileMap block_2_ctile_map_;
        index_t M01_;
        index_t N01_;
        AElementwiseOperation a_element_op_;
        BElementwiseOperation b_element_op_;
        CElementwiseOperation c_element_op_;
    };

    // Invoker
    struct Invoker : public BaseInvoker
    {
        using Argument = DeviceBatchedGemmTransposeXdl::Argument;

        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            {
                std::cout << "arg.a_grid_desc_k0_m_k1_{" << arg.a_grid_desc_k0_m_k1_.GetLength(I0)
                          << ", " << arg.a_grid_desc_k0_m_k1_.GetLength(I1) << ", "
                          << arg.a_grid_desc_k0_m_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.b_grid_desc_k0_n_k1_{" << arg.b_grid_desc_k0_n_k1_.GetLength(I0)
                          << ", " << arg.b_grid_desc_k0_n_k1_.GetLength(I1) << ", "
                          << arg.b_grid_desc_k0_n_k1_.GetLength(I2) << "}" << std::endl;

                std::cout << "arg.c_grid_desc_m_n_{" << arg.c_grid_desc_m_n_.GetLength(I0) << ", "
                          << arg.c_grid_desc_m_n_.GetLength(I1) << "}" << std::endl;
            }

            if(!GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                            arg.b_grid_desc_k0_n_k1_,
                                            arg.c_grid_desc_m_n_,
                                            arg.block_2_ctile_map_))
            {
                throw std::runtime_error(
                    "wrong! GridwiseBatchedGemmTranspose_km_kn_m0m1n0n1_xdlops_v2r3 has invalid "
                    "setting");
            }

            const index_t grid_size =
                arg.block_2_ctile_map_.CalculateGridSize(arg.c_grid_desc_m_n_) * arg.BatchCount_;

            const auto K =
                arg.a_grid_desc_k0_m_k1_.GetLength(I0) * arg.a_grid_desc_k0_m_k1_.GetLength(I2);

            float ave_time = 0;

            if(GridwiseGemm::CalculateHasMainKBlockLoop(K))
            {
                const auto kernel = kernel_batched_gemm_transpose_xdlops_v2r3<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceBatchedGemmTransposeXdl::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceBatchedGemmTransposeXdl::BGridDesc_K0_N_K1>,
                    remove_reference_t<typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    ComputePtrOffsetOfStridedBatch,
                    remove_reference_t<Block2CTileMap>,
                    true>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.BatchCount_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_,
                                                  arg.compute_ptr_offset_of_batch_,
                                                  arg.block_2_ctile_map_);
            }
            else
            {
                const auto kernel = kernel_batched_gemm_transpose_xdlops_v2r3<
                    GridwiseGemm,
                    ADataType, // TODO: distiguish A/B datatype
                    CDataType,
                    remove_reference_t<DeviceBatchedGemmTransposeXdl::AGridDesc_K0_M_K1>,
                    remove_reference_t<DeviceBatchedGemmTransposeXdl::BGridDesc_K0_N_K1>,
                    remove_reference_t<typename GridwiseGemm::CGridDesc_M0_N0_M1_N1_M2_M3_M4_N2>,
                    AElementwiseOperation,
                    BElementwiseOperation,
                    CElementwiseOperation,
                    ComputePtrOffsetOfStridedBatch,
                    remove_reference_t<Block2CTileMap>,
                    false>;

                ave_time = launch_and_time_kernel(stream_config,
                                                  kernel,
                                                  dim3(grid_size),
                                                  dim3(BlockSize),
                                                  0,
                                                  arg.p_a_grid_,
                                                  arg.p_b_grid_,
                                                  arg.p_c_grid_,
                                                  arg.BatchCount_,
                                                  arg.a_grid_desc_k0_m_k1_,
                                                  arg.b_grid_desc_k0_n_k1_,
                                                  arg.c_grid_desc_m0_n0_m1_n1_m2_m3_m4_n2_,
                                                  arg.a_element_op_,
                                                  arg.b_element_op_,
                                                  arg.c_element_op_,
                                                  arg.compute_ptr_offset_of_batch_,
                                                  arg.block_2_ctile_map_);
            }

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        return GridwiseGemm::CheckValidity(arg.a_grid_desc_k0_m_k1_,
                                           arg.b_grid_desc_k0_n_k1_,
                                           arg.c_grid_desc_m_n_,
                                           arg.block_2_ctile_map_);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const ADataType* p_a,
                             const BDataType* p_b,
                             CDataType* p_c,
                             GemmTransposeDesc gemm_transpose_desc,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op,
                             index_t BatchCount)
    {
        return Argument{p_a,
                        p_b,
                        p_c,
                        gemm_transpose_desc,
                        1,
                        1,
                        a_element_op,
                        b_element_op,
                        c_element_op,
                        BatchCount};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      void* p_c,
                                                      GemmTransposeDesc gemm_transpose_desc,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op,
                                                      index_t BatchCount) override
    {
        return std::make_unique<Argument>(static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          static_cast<CDataType*>(p_c),
                                          gemm_transpose_desc,
                                          1,
                                          1,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op,
                                          BatchCount);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        // clang-format off
        str << "DeviceBatchedGemmTransposeXdl"
            << "<"
            << BlockSize << ", "
            << MPerBlock << ", "
            << NPerBlock << ", "
            << K0PerBlock
            << ">";
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
#endif
