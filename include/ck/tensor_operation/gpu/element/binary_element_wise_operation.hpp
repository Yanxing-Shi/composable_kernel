#pragma once
#include "data_type.hpp"

namespace ck {
namespace tensor_operation {
namespace binary_element_wise {

struct Add
{
    __host__ __device__ constexpr void
    operator()(double& dst, const double& src1, const double& src2) const
    {
        dst = src1 + src2;
    }

    __host__ __device__ constexpr void
    operator()(float& dst, const float& src1, const float& src2) const
    {
        dst = src1 + src2;
    }

    __host__ __device__ constexpr void
    operator()(half_t& dst, const half_t& src1, const half_t& src2) const
    {
        dst = src1 + src2;
    }

    __host__ __device__ constexpr void
    operator()(bhalf_t& dst, const bhalf_t& src1, const bhalf_t& src2) const
    {
        const float x1 = ck::type_convert<float>(src1);
        const float x2 = ck::type_convert<float>(src2);
        const float y  = x1 + x2;
        dst            = ck::type_convert<bhalf_t>(y);
    }
};

struct Substract
{
    __host__ __device__ constexpr void
    operator()(double& dst, const double& src1, const double& src2) const
    {
        dst = src1 - src2;
    }

    __host__ __device__ constexpr void
    operator()(float& dst, const float& src1, const float& src2) const
    {
        dst = src1 - src2;
    }

    __host__ __device__ constexpr void
    operator()(half_t& dst, const half_t& src1, const half_t& src2) const
    {
        dst = src1 - src2;
    }

    __host__ __device__ constexpr void
    operator()(bhalf_t& dst, const bhalf_t& src1, const bhalf_t& src2) const
    {
        const float x1 = ck::type_convert<float>(src1);
        const float x2 = ck::type_convert<float>(src2);
        const float y  = x1 - x2;
        dst            = ck::type_convert<bhalf_t>(y);
    }
};

} // namespace binary_element_wise
} // namespace tensor_operation
} // namespace ck
