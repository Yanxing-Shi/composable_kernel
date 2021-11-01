#pragma once
#include "device_gemm_xdl_instance.hpp"

namespace ck {
namespace profiler {
namespace device_gemm_xdl_instance {

void add_device_gemms_xdl_instnace_f16_f16_f16_mk_nk_mn_vol1(
    std::vector<DeviceOpCombo>& device_op_combos,
    ck::half_t* p_a,
    ck::half_t* p_b,
    ck::half_t* p_c,
    int M,
    int N,
    int K,
    int StrideA,
    int StrideB,
    int StrideC);

void add_device_gemms_xdl_instnace_f16_f16_f16_mk_nk_mn_vol2(
    std::vector<DeviceOpCombo>& device_op_combos,
    ck::half_t* p_a,
    ck::half_t* p_b,
    ck::half_t* p_c,
    int M,
    int N,
    int K,
    int StrideA,
    int StrideB,
    int StrideC);

} // namespace device_gemm_xdl_instance
} // namespace profiler
} // namespace ck

namespace ck {
namespace profiler {

#if 0
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout,
          typename BLayout,
          typename CLayout>
#endif
void profile_gemm(int argc, char* argv[])
{
    using namespace ck;

    // Currently ADataType and BDataType need to be the same
    using ADataType   = half_t;
    using BDataType   = half_t;
    using CDataType   = half_t;
    using AccDataType = float;

    // NT problem
    using ALayout = tensor_layout::RowMajor;
    using BLayout = tensor_layout::ColumnMajor;
    using CLayout = tensor_layout::RowMajor;

    if(argc != 11)
    {
        printf("arg1 to 4: do_verification, init_method, do_log, nrepeat\n");
        printf("arg5 to 10: M, N, K, StrideA, StrideB, StrideC\n");
        exit(1);
    }

    const bool do_verification = std::stoi(argv[1]);
    const int init_method      = std::stoi(argv[2]);
    const bool do_log          = std::stoi(argv[3]);
    const int nrepeat          = std::stoi(argv[4]);

    const index_t M = std::stoi(argv[5]);
    const index_t N = std::stoi(argv[6]);
    const index_t K = std::stoi(argv[7]);

    const index_t StrideA = std::stoi(argv[8]);
    const index_t StrideB = std::stoi(argv[9]);
    const index_t StrideC = std::stoi(argv[10]);

    auto f_host_tensor_descriptor =
        [](std::size_t row, std::size_t col, std::size_t stride, auto layout) {
            if(is_same<decltype(layout), tensor_layout::RowMajor>::value)
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({stride, 1}));
            }
            else
            {
                return HostTensorDescriptor(std::vector<std::size_t>({row, col}),
                                            std::vector<std::size_t>({1, stride}));
            }
        };

    Tensor<ADataType> a_m_k(f_host_tensor_descriptor(M, K, StrideA, ALayout{}));
    Tensor<BDataType> b_k_n(f_host_tensor_descriptor(K, N, StrideB, BLayout{}));
    Tensor<BDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
    Tensor<BDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));

    std::cout << "a_m_k: " << a_m_k.mDesc << std::endl;
    std::cout << "b_k_n: " << b_k_n.mDesc << std::endl;
    std::cout << "c_m_n: " << c_m_n_host_result.mDesc << std::endl;

    switch(init_method)
    {
    case 0:
        // no initialization
        break;
    case 1:
        a_m_k.GenerateTensorValue(GeneratorTensor_1{});
        b_k_n.GenerateTensorValue(GeneratorTensor_1{});
        break;
    case 2:
        a_m_k.GenerateTensorValue(GeneratorTensor_1{});
        b_k_n.GenerateTensorValue(GeneratorTensor_2{-5, 5});
        break;
    case 3:
        a_m_k.GenerateTensorValue(GeneratorTensor_2{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_1{});
        break;
    case 4:
        a_m_k.GenerateTensorValue(GeneratorTensor_2{-5, 5});
        b_k_n.GenerateTensorValue(GeneratorTensor_2{-5, 5});
        break;
    default:
        a_m_k.GenerateTensorValue(GeneratorTensor_3<float>{0.0, 1.0});
        b_k_n.GenerateTensorValue(GeneratorTensor_3<float>{-0.5, 0.5});
    }

    DeviceMem a_device_buf(sizeof(ADataType) * a_m_k.mDesc.GetElementSpace());
    DeviceMem b_device_buf(sizeof(BDataType) * b_k_n.mDesc.GetElementSpace());
    DeviceMem c_device_buf(sizeof(CDataType) * c_m_n_device_result.mDesc.GetElementSpace());

    a_device_buf.ToDevice(a_m_k.mData.data());
    b_device_buf.ToDevice(b_k_n.mData.data());
    c_device_buf.ToDevice(c_m_n_device_result.mData.data());

    std::vector<DeviceOpCombo> device_gemm_combos;

    if constexpr(is_same_v<ALayout, tensor_layout::RowMajor> &&
                 is_same_v<BLayout, tensor_layout::ColumnMajor> &&
                 is_same_v<CLayout, tensor_layout::RowMajor>)
    {
        device_gemm_xdl_instance::add_device_gemms_xdl_instnace_f16_f16_f16_mk_nk_mn_vol1(
            device_gemm_combos,
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC);

        device_gemm_xdl_instance::add_device_gemms_xdl_instnace_f16_f16_f16_mk_nk_mn_vol2(
            device_gemm_combos,
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC);
    }
#if 0
    else if constexpr(is_same_v<ALayout, tensor_layout::ColumnMajor> &&
                      is_same_v<BLayout, tensor_layout::RowMajor> &&
                      is_same_v<CLayout, tensor_layout::RowMajor>)
    {
        add_device_gemms_xdl_f16_f16_f16_km_kn_mn(
            device_gemm_combos,
            static_cast<ADataType*>(a_device_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_device_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(c_device_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            StrideA,
            StrideB,
            StrideC);
    }
#endif

    for(auto& device_gemm_combo : device_gemm_combos)
    {
        auto& gemm_ptr     = std::get<0>(device_gemm_combo);
        auto& invoker_ptr  = std::get<1>(device_gemm_combo);
        auto& argument_ptr = std::get<2>(device_gemm_combo);

        if(!gemm_ptr->IsSupportedArgument(argument_ptr.get()))
        {
            throw std::runtime_error(
                "wrong! device_gemm with the specified compilation parameters does "
                "not support this GEMM problem");
        }

        for(int i = 0; i < 5; ++i)
        {
            float ave_time = invoker_ptr->Run(argument_ptr.get(), nrepeat);

            std::size_t flop = std::size_t(2) * M * N * K;
            std::size_t num_btype =
                sizeof(ADataType) * M * K + sizeof(BDataType) * K * M + sizeof(CDataType) * M * N;

            float tflops = static_cast<float>(flop) / 1.E9 / ave_time;

            float gb_per_sec = num_btype / 1.E6 / ave_time;

            std::cout << "Perf: " << ave_time << " ms, " << tflops << " TFlops, " << gb_per_sec
                      << " GB/s" << std::endl;
        }
    }

    // copy result back to host
    c_device_buf.FromDevice(c_m_n_device_result.mData.data());

    if(do_verification)
    {
        host_gemm_mk_kn_mn(a_m_k, b_k_n, c_m_n_host_result);

        check_error(c_m_n_host_result, c_m_n_device_result);

        if(do_log)
        {
            LogRangeAsType<float>(std::cout << "a : ", a_m_k.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "b: ", b_k_n.mData, ",") << std::endl;
            LogRangeAsType<float>(std::cout << "c_host  : ", c_m_n_host_result.mData, ",")
                << std::endl;
            LogRangeAsType<float>(std::cout << "c_device: ", c_m_n_device_result.mData, ",")
                << std::endl;
        }
    }
}

} // namespace profiler
} // namespace ck
