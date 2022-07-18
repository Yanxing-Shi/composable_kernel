// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <vector>

#include "device_base.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
struct DeviceBatchedGemmMultiD : public BaseOperator
{
    static constexpr index_t NumDTensor = DsDataType::Size();

    virtual std::unique_ptr<BaseArgument>
    MakeArgumentPointer(const void* p_a,
                        const void* p_b,
                        void* p_c,
                        ck::index_t M,
                        ck::index_t N,
                        ck::index_t K,
                        ck::index_t StrideA,
                        ck::index_t StrideB,
                        //std::array<ck::index_t, NumDTensor> StrideDs,
                        ck::index_t StrideE,
                        ck::index_t BatchStrideA,
                        ck::index_t BatchStrideB,
                        ck::index_t BatchStrideE,
                        ck::index_t Batch,
                        AElementwiseOperation a_element_op,
                        BElementwiseOperation b_element_op,
                        CDEElementwiseOperation cde_element_op) = 0;

    virtual std::unique_ptr<BaseInvoker> MakeInvokerPointer() = 0;
};

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename EDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CDEElementwiseOperation>
using DeviceBatchedGemmMultiDPtr = std::unique_ptr<DeviceBatchedGemmMultiD<ALayout,
                                                               BLayout,
                                                               CLayout,
                                                               ADataType,
                                                               BDataType,
                                                               DsDataType,
                                                               EDataType,
                                                               AElementwiseOperation,
                                                               BElementwiseOperation,
                                                               CDEElementwiseOperation>>;

} // namespace device
} // namespace tensor_operation
} // namespace ck
