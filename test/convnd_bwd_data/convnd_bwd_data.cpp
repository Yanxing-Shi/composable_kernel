// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>
#include <vector>

#include "profiler/include/profile_conv_bwd_data_impl.hpp"

int main()
{
    bool pass = true;

    std::vector<ck::tensor_operation::device::ConvParams> params;

    // check 1d
    params.push_back({1, 128, 128, 256, {1}, {14}, {2}, {1}, {0}, {0}});
    params.push_back({1, 128, 128, 256, {3}, {28}, {1}, {1}, {1}, {1}});
    params.push_back({1, 128, 128, 256, {1}, {3}, {1}, {1}, {0}, {0}});

    for(auto& param : params)
    {
        pass &= ck::profiler::profile_conv_bwd_data_impl<1,
                                                         ck::tensor_layout::convolution::NWC,
                                                         ck::tensor_layout::convolution::KXC,
                                                         ck::tensor_layout::convolution::NWK,
                                                         float,
                                                         float,
                                                         float>(true,  // do_verification
                                                                1,     // init_method
                                                                false, // do_log
                                                                false, // time_kernel
                                                                param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<1,
                                                         ck::tensor_layout::convolution::NWC,
                                                         ck::tensor_layout::convolution::KXC,
                                                         ck::tensor_layout::convolution::NWK,
                                                         ck::half_t,
                                                         ck::half_t,
                                                         ck::half_t>(true,  // do_verification
                                                                     1,     // init_method
                                                                     false, // do_log
                                                                     false, // time_kernel
                                                                     param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<1,
                                                         ck::tensor_layout::convolution::NWC,
                                                         ck::tensor_layout::convolution::KXC,
                                                         ck::tensor_layout::convolution::NWK,
                                                         ck::bhalf_t,
                                                         ck::bhalf_t,
                                                         ck::bhalf_t>(true,  // do_verification
                                                                      1,     // init_method
                                                                      false, // do_log
                                                                      false, // time_kernel
                                                                      param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<1,
                                                         ck::tensor_layout::convolution::NWC,
                                                         ck::tensor_layout::convolution::KXC,
                                                         ck::tensor_layout::convolution::NWK,
                                                         int8_t,
                                                         int8_t,
                                                         int8_t>(true,  // do_verification
                                                                 1,     // init_method
                                                                 false, // do_log
                                                                 false, // time_kernel
                                                                 param);
    }

    // check 2d
    params.clear();
    params.push_back({2, 128, 128, 256, {1, 1}, {7, 7}, {2, 2}, {1, 1}, {0, 0}, {0, 0}});
    params.push_back({2, 128, 128, 256, {3, 3}, {14, 14}, {1, 1}, {1, 1}, {1, 1}, {1, 1}});
    params.push_back({2, 128, 128, 256, {1, 1}, {3, 3}, {1, 1}, {1, 1}, {0, 0}, {0, 0}});

    for(auto& param : params)
    {
        pass &= ck::profiler::profile_conv_bwd_data_impl<2,
                                                         ck::tensor_layout::convolution::NHWC,
                                                         ck::tensor_layout::convolution::KYXC,
                                                         ck::tensor_layout::convolution::NHWK,
                                                         float,
                                                         float,
                                                         float>(true,  // do_verification
                                                                1,     // init_method
                                                                false, // do_log
                                                                false, // time_kernel
                                                                param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<2,
                                                         ck::tensor_layout::convolution::NHWC,
                                                         ck::tensor_layout::convolution::KYXC,
                                                         ck::tensor_layout::convolution::NHWK,
                                                         ck::half_t,
                                                         ck::half_t,
                                                         ck::half_t>(true,  // do_verification
                                                                     1,     // init_method
                                                                     false, // do_log
                                                                     false, // time_kernel
                                                                     param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<2,
                                                         ck::tensor_layout::convolution::NHWC,
                                                         ck::tensor_layout::convolution::KYXC,
                                                         ck::tensor_layout::convolution::NHWK,
                                                         ck::bhalf_t,
                                                         ck::bhalf_t,
                                                         ck::bhalf_t>(true,  // do_verification
                                                                      1,     // init_method
                                                                      false, // do_log
                                                                      false, // time_kernel
                                                                      param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<2,
                                                         ck::tensor_layout::convolution::NHWC,
                                                         ck::tensor_layout::convolution::KYXC,
                                                         ck::tensor_layout::convolution::NHWK,
                                                         int8_t,
                                                         int8_t,
                                                         int8_t>(true,  // do_verification
                                                                 1,     // init_method
                                                                 false, // do_log
                                                                 false, // time_kernel
                                                                 param);
    }

    // check 3d
    params.clear();
    params.push_back(
        {3, 128, 128, 256, {1, 1, 1}, {7, 7, 7}, {2, 2, 2}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});
    params.push_back(
        {3, 128, 128, 256, {3, 3, 3}, {14, 14, 14}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}});
    params.push_back(
        {3, 128, 128, 256, {1, 1, 1}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}});

    for(auto& param : params)
    {
        pass &= ck::profiler::profile_conv_bwd_data_impl<3,
                                                         ck::tensor_layout::convolution::NDHWC,
                                                         ck::tensor_layout::convolution::KZYXC,
                                                         ck::tensor_layout::convolution::NDHWK,
                                                         float,
                                                         float,
                                                         float>(true,  // do_verification
                                                                1,     // init_method
                                                                false, // do_log
                                                                false, // time_kernel
                                                                param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<3,
                                                         ck::tensor_layout::convolution::NDHWC,
                                                         ck::tensor_layout::convolution::KZYXC,
                                                         ck::tensor_layout::convolution::NDHWK,
                                                         ck::half_t,
                                                         ck::half_t,
                                                         ck::half_t>(true,  // do_verification
                                                                     1,     // init_method
                                                                     false, // do_log
                                                                     false, // time_kernel
                                                                     param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<3,
                                                         ck::tensor_layout::convolution::NDHWC,
                                                         ck::tensor_layout::convolution::KZYXC,
                                                         ck::tensor_layout::convolution::NDHWK,
                                                         ck::bhalf_t,
                                                         ck::bhalf_t,
                                                         ck::bhalf_t>(true,  // do_verification
                                                                      1,     // init_method
                                                                      false, // do_log
                                                                      false, // time_kernel
                                                                      param);

        pass &= ck::profiler::profile_conv_bwd_data_impl<3,
                                                         ck::tensor_layout::convolution::NDHWC,
                                                         ck::tensor_layout::convolution::KZYXC,
                                                         ck::tensor_layout::convolution::NDHWK,
                                                         int8_t,
                                                         int8_t,
                                                         int8_t>(true,  // do_verification
                                                                 1,     // init_method
                                                                 false, // do_log
                                                                 false, // time_kernel
                                                                 param);
    }

    if(pass)
    {
        std::cout << "test convnd bwd : Pass" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "test convnd bwd: Fail " << std::endl;
        return -1;
    }
}
