/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include "concepts.hpp"
#include "helpers.hpp"

#include "hc.hpp"
#include "hip/hip_ext.h"
#include "hip_runtime.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hip_impl {
namespace {
struct New_grid_launch_tag {};
struct Old_grid_launch_tag {};

template <typename C, typename D>
class RAII_guard {
    D dtor_;

   public:
    RAII_guard() = default;

    RAII_guard(const C& ctor, D dtor) : dtor_{std::move(dtor)} { ctor(); }

    RAII_guard(const RAII_guard&) = default;
    RAII_guard(RAII_guard&&) = default;

    RAII_guard& operator=(const RAII_guard&) = default;
    RAII_guard& operator=(RAII_guard&&) = default;

    ~RAII_guard() { dtor_(); }
};

template <typename C, typename D>
RAII_guard<C, D> make_RAII_guard(const C& ctor, D dtor) {
    return RAII_guard<C, D>{ctor, std::move(dtor)};
}

template <FunctionalProcedure F, typename... Ts>
using is_new_grid_launch_t = typename std::conditional<is_callable<F(Ts...)>{}, New_grid_launch_tag,
                                                       Old_grid_launch_tag>::type;
}  // namespace

// TODO: - dispatch rank should be derived from the domain dimensions passed
//         in, and not always assumed to be 3;

template <FunctionalProcedure K, typename... Ts>
requires(Domain<K> ==
         {Ts...}) inline void grid_launch_hip_impl_(New_grid_launch_tag, dim3 num_blocks,
                                                    dim3 dim_blocks, int group_mem_bytes,
                                                    const hc::accelerator_view& acc_v, K k) {
    const auto d =
        hc::extent<3>{num_blocks.z * dim_blocks.z, num_blocks.y * dim_blocks.y,
                      num_blocks.x * dim_blocks.x}
            .tile_with_dynamic(dim_blocks.z, dim_blocks.y, dim_blocks.x, group_mem_bytes);

    try {
        hc::parallel_for_each(acc_v, d, k);
    } catch (std::exception& ex) {
        std::cerr << "Failed in " << __func__ << ", with exception: " << ex.what() << std::endl;
        hip_throw(ex);
    }
}

// TODO: these are workarounds, they should be removed.

hc::accelerator_view lock_stream_hip_(hipStream_t&, void*&);
void print_prelaunch_trace_(const char*, dim3, dim3, int, hipStream_t);
void unlock_stream_hip_(hipStream_t, void*, const char*, hc::accelerator_view*);

template <FunctionalProcedure K, typename... Ts>
requires(Domain<K> == {Ts...}) inline void grid_launch_hip_impl_(New_grid_launch_tag,
                                                                 dim3 num_blocks, dim3 dim_blocks,
                                                                 int group_mem_bytes,
                                                                 hipStream_t stream,
                                                                 const char* kernel_name, K k) {
    void* lck_stream = nullptr;
    auto acc_v = lock_stream_hip_(stream, lck_stream);
    auto stream_guard =
        make_RAII_guard(std::bind(print_prelaunch_trace_, kernel_name, num_blocks, dim_blocks,
                                  group_mem_bytes, stream),
                        std::bind(unlock_stream_hip_, stream, lck_stream, kernel_name, &acc_v));

    try {
        grid_launch_hip_impl_(New_grid_launch_tag{}, std::move(num_blocks), std::move(dim_blocks),
                              group_mem_bytes, acc_v, std::move(k));
    } catch (std::exception& ex) {
        std::cerr << "Failed in " << __func__ << ", with exception: " << ex.what() << std::endl;
        hip_throw(ex);
    }
}

template <FunctionalProcedure K, typename... Ts>
requires(Domain<K> ==
         {hipLaunchParm, Ts...}) inline void grid_launch_hip_impl_(Old_grid_launch_tag,
                                                                   dim3 num_blocks, dim3 dim_blocks,
                                                                   int group_mem_bytes,
                                                                   hipStream_t stream, K k) {
    grid_launch_hip_impl_(New_grid_launch_tag{}, std::move(num_blocks), std::move(dim_blocks),
                          group_mem_bytes, std::move(stream), std::move(k));
}

template <FunctionalProcedure K, typename... Ts>
requires(Domain<K> == {hipLaunchParm, Ts...}) inline void grid_launch_hip_impl_(
    Old_grid_launch_tag, dim3 num_blocks, dim3 dim_blocks, int group_mem_bytes, hipStream_t stream,
    const char* kernel_name, K k) {
    grid_launch_hip_impl_(New_grid_launch_tag{}, std::move(num_blocks), std::move(dim_blocks),
                          group_mem_bytes, std::move(stream), kernel_name, std::move(k));
}

template <FunctionalProcedure K, typename... Ts>
requires(Domain<K> == {Ts...}) inline std::enable_if_t<
    !std::is_function<K>::value> grid_launch_hip_(dim3 num_blocks, dim3 dim_blocks,
                                                  int group_mem_bytes, hipStream_t stream,
                                                  const char* kernel_name, K k) {
    grid_launch_hip_impl_(is_new_grid_launch_t<K, Ts...>{}, std::move(num_blocks),
                          std::move(dim_blocks), group_mem_bytes, std::move(stream), kernel_name,
                          std::move(k));
}

template <FunctionalProcedure K, typename... Ts>
requires(Domain<K> == {Ts...}) inline std::enable_if_t<
    !std::is_function<K>::value> grid_launch_hip_(dim3 num_blocks, dim3 dim_blocks,
                                                  int group_mem_bytes, hipStream_t stream, K k) {
    grid_launch_hip_impl_(is_new_grid_launch_t<K, Ts...>{}, std::move(num_blocks),
                          std::move(dim_blocks), group_mem_bytes, std::move(stream), std::move(k));
}

// TODO: these are temporary and purposefully noisy and disruptive.
#define make_kernel_name_hip(k, n)                                                                 \
    HIP_kernel_functor_name_begin##_##k##_##HIP_kernel_functor_name_end##_##n

#define make_kernel_functor_hip_30(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, \
                                   p22, p23, p24, p25, p26, p27)                                   \
    struct make_kernel_name_hip(function_name, 28) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        std::decay_t<decltype(p20)> _p20_;                                                         \
        std::decay_t<decltype(p21)> _p21_;                                                         \
        std::decay_t<decltype(p22)> _p22_;                                                         \
        std::decay_t<decltype(p23)> _p23_;                                                         \
        std::decay_t<decltype(p24)> _p24_;                                                         \
        std::decay_t<decltype(p25)> _p25_;                                                         \
        std::decay_t<decltype(p26)> _p26_;                                                         \
        std::decay_t<decltype(p27)> _p27_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_, _p20_, _p21_,      \
                        _p22_, _p23_, _p24_, _p25_, _p26_, _p27_);                                 \
        }                                                                                          \
    }
#define make_kernel_functor_hip_29(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, \
                                   p22, p23, p24, p25, p26)                                        \
    struct make_kernel_name_hip(function_name, 27) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        std::decay_t<decltype(p20)> _p20_;                                                         \
        std::decay_t<decltype(p21)> _p21_;                                                         \
        std::decay_t<decltype(p22)> _p22_;                                                         \
        std::decay_t<decltype(p23)> _p23_;                                                         \
        std::decay_t<decltype(p24)> _p24_;                                                         \
        std::decay_t<decltype(p25)> _p25_;                                                         \
        std::decay_t<decltype(p26)> _p26_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_, _p20_, _p21_,      \
                        _p22_, _p23_, _p24_, _p25_, _p26_);                                        \
        }                                                                                          \
    }
#define make_kernel_functor_hip_28(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, \
                                   p22, p23, p24, p25)                                             \
    struct make_kernel_name_hip(function_name, 26) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        std::decay_t<decltype(p20)> _p20_;                                                         \
        std::decay_t<decltype(p21)> _p21_;                                                         \
        std::decay_t<decltype(p22)> _p22_;                                                         \
        std::decay_t<decltype(p23)> _p23_;                                                         \
        std::decay_t<decltype(p24)> _p24_;                                                         \
        std::decay_t<decltype(p25)> _p25_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_, _p20_, _p21_,      \
                        _p22_, _p23_, _p24_, _p25_);                                               \
        }                                                                                          \
    }
#define make_kernel_functor_hip_27(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, \
                                   p22, p23, p24)                                                  \
    struct make_kernel_name_hip(function_name, 25) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        std::decay_t<decltype(p20)> _p20_;                                                         \
        std::decay_t<decltype(p21)> _p21_;                                                         \
        std::decay_t<decltype(p22)> _p22_;                                                         \
        std::decay_t<decltype(p23)> _p23_;                                                         \
        std::decay_t<decltype(p24)> _p24_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_, _p20_, _p21_,      \
                        _p22_, _p23_, _p24_);                                                      \
        }                                                                                          \
    }
#define make_kernel_functor_hip_26(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, \
                                   p22, p23)                                                       \
    struct make_kernel_name_hip(function_name, 24) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        std::decay_t<decltype(p20)> _p20_;                                                         \
        std::decay_t<decltype(p21)> _p21_;                                                         \
        std::decay_t<decltype(p22)> _p22_;                                                         \
        std::decay_t<decltype(p23)> _p23_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_, _p20_, _p21_,      \
                        _p22_, _p23_);                                                             \
        }                                                                                          \
    }
#define make_kernel_functor_hip_25(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, \
                                   p22)                                                            \
    struct make_kernel_name_hip(function_name, 23) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        std::decay_t<decltype(p20)> _p20_;                                                         \
        std::decay_t<decltype(p21)> _p21_;                                                         \
        std::decay_t<decltype(p22)> _p22_;                                                         \
        __attribute__((used, flatten)) void operator()(const hc::tiled_index<3>&) const [[hc]] {   \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_, _p20_, _p21_,      \
                        _p22_);                                                                    \
        }                                                                                          \
    }
#define make_kernel_functor_hip_24(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21) \
    struct make_kernel_name_hip(function_name, 22) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        std::decay_t<decltype(p20)> _p20_;                                                         \
        std::decay_t<decltype(p21)> _p21_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_, _p20_, _p21_);     \
        }                                                                                          \
    }
#define make_kernel_functor_hip_23(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20)      \
    struct make_kernel_name_hip(function_name, 21) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        std::decay_t<decltype(p20)> _p20_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_, _p20_);            \
        }                                                                                          \
    }
#define make_kernel_functor_hip_22(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19)           \
    struct make_kernel_name_hip(function_name, 20) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        std::decay_t<decltype(p19)> _p19_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_, _p19_);                   \
        }                                                                                          \
    }
#define make_kernel_functor_hip_21(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17, p18)                \
    struct make_kernel_name_hip(function_name, 19) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        std::decay_t<decltype(p18)> _p18_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_, _p18_);                          \
        }                                                                                          \
    }
#define make_kernel_functor_hip_20(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16, p17)                     \
    struct make_kernel_name_hip(function_name, 18) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        std::decay_t<decltype(p17)> _p17_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_, _p17_);                                 \
        }                                                                                          \
    }
#define make_kernel_functor_hip_19(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15, p16)                          \
    struct make_kernel_name_hip(function_name, 17) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        std::decay_t<decltype(p16)> _p16_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_, _p16_);                                        \
        }                                                                                          \
    }
#define make_kernel_functor_hip_18(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14, p15)                               \
    struct make_kernel_name_hip(function_name, 16) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        std::decay_t<decltype(p15)> _p15_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_, _p15_);                                               \
        }                                                                                          \
    }
#define make_kernel_functor_hip_17(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13, p14)                                    \
    struct make_kernel_name_hip(function_name, 15) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        std::decay_t<decltype(p14)> _p14_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_, _p14_);                                                      \
        }                                                                                          \
    }
#define make_kernel_functor_hip_16(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12, p13)                                         \
    struct make_kernel_name_hip(function_name, 14) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        std::decay_t<decltype(p13)> _p13_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_, _p13_);                                                             \
        }                                                                                          \
    }
#define make_kernel_functor_hip_15(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11, p12)                                              \
    struct make_kernel_name_hip(function_name, 13) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        std::decay_t<decltype(p12)> _p12_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_,  \
                        _p12_);                                                                    \
        }                                                                                          \
    }
#define make_kernel_functor_hip_14(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10, p11)                                                   \
    struct make_kernel_name_hip(function_name, 12) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        std::decay_t<decltype(p11)> _p11_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_, _p11_); \
        }                                                                                          \
    }
#define make_kernel_functor_hip_13(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9, p10)                                                        \
    struct make_kernel_name_hip(function_name, 11) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        std::decay_t<decltype(p10)> _p10_;                                                         \
        void operator()(const hc::tiled_index<3>&) const [[hc]] {                                  \
            kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_, _p10_);        \
        }                                                                                          \
    }
#define make_kernel_functor_hip_12(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8, \
                                   p9)                                                             \
    struct make_kernel_name_hip(function_name, 10) {                                               \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        std::decay_t<decltype(p9)> _p9_;                                                           \
        void operator()(const hc::tiled_index<3>&) const                                           \
            [[hc]] { kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_, _p9_); }    \
    }
#define make_kernel_functor_hip_11(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7, p8) \
    struct make_kernel_name_hip(function_name, 9) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        std::decay_t<decltype(p8)> _p8_;                                                           \
        void operator()(const hc::tiled_index<3>&) const                                           \
            [[hc]] { kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_, _p8_); }          \
    }
#define make_kernel_functor_hip_10(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6, p7)     \
    struct make_kernel_name_hip(function_name, 8) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        std::decay_t<decltype(p7)> _p7_;                                                           \
        void operator()(const hc::tiled_index<3>&) const                                           \
            [[hc]] { kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_, _p7_); }                \
    }
#define make_kernel_functor_hip_9(function_name, kernel_name, p0, p1, p2, p3, p4, p5, p6)          \
    struct make_kernel_name_hip(function_name, 7) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        std::decay_t<decltype(p6)> _p6_;                                                           \
        void operator()(const hc::tiled_index<3>&) const                                           \
            [[hc]] { kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_, _p6_); }                      \
    }
#define make_kernel_functor_hip_8(function_name, kernel_name, p0, p1, p2, p3, p4, p5)              \
    struct make_kernel_name_hip(function_name, 6) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        std::decay_t<decltype(p5)> _p5_;                                                           \
        void operator()(const hc::tiled_index<3>&) const                                           \
            [[hc]] { kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_, _p5_); }                            \
    }
#define make_kernel_functor_hip_7(function_name, kernel_name, p0, p1, p2, p3, p4)                  \
    struct make_kernel_name_hip(function_name, 5) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        std::decay_t<decltype(p4)> _p4_;                                                           \
        void operator()(const hc::tiled_index<3>&) const                                           \
            [[hc]] { kernel_name(_p0_, _p1_, _p2_, _p3_, _p4_); }                                  \
    }
#define make_kernel_functor_hip_6(function_name, kernel_name, p0, p1, p2, p3)                      \
    struct make_kernel_name_hip(function_name, 4) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        std::decay_t<decltype(p3)> _p3_;                                                           \
        void operator()(const hc::tiled_index<3>&) const                                           \
            [[hc]] { kernel_name(_p0_, _p1_, _p2_, _p3_); }                                        \
    }
#define make_kernel_functor_hip_5(function_name, kernel_name, p0, p1, p2)                          \
    struct make_kernel_name_hip(function_name, 3) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        std::decay_t<decltype(p2)> _p2_;                                                           \
        void operator()(const hc::tiled_index<3>&) const [[hc]] { kernel_name(_p0_, _p1_, _p2_); } \
    }
#define make_kernel_functor_hip_4(function_name, kernel_name, p0, p1)                              \
    struct make_kernel_name_hip(function_name, 2) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        std::decay_t<decltype(p1)> _p1_;                                                           \
        void operator()(const hc::tiled_index<3>&) const [[hc]] { kernel_name(_p0_, _p1_); }       \
    }
#define fofo(f, n) kernel_prefix_hip##f##kernel_suffix_hip##n
#define make_kernel_functor_hip_3(function_name, kernel_name, p0)                                  \
    struct make_kernel_name_hip(function_name, 1) {                                                \
        std::decay_t<decltype(p0)> _p0_;                                                           \
        void operator()(const hc::tiled_index<3>&) const [[hc]] { kernel_name(_p0_); }             \
    }
#define make_kernel_functor_hip_2(function_name, kernel_name)                                      \
    struct make_kernel_name_hip(function_name, 0) {                                                \
        void operator()(const hc::tiled_index<3>&)[[hc]] { return kernel_name(hipLaunchParm{}); }  \
    }
#define make_kernel_functor_hip_1(...)
#define make_kernel_functor_hip_0(...)
#define make_kernel_functor_hip_(...) overload_macro_hip_(make_kernel_functor_hip_, __VA_ARGS__)


#define hipLaunchNamedKernelGGL(function_name, kernel_name, num_blocks, dim_blocks,                \
                                group_mem_bytes, stream, ...)                                      \
    do {                                                                                           \
        make_kernel_functor_hip_(function_name, kernel_name, __VA_ARGS__)                          \
            hip_kernel_functor_impl_{__VA_ARGS__};                                                 \
        hip_impl::grid_launch_hip_(num_blocks, dim_blocks, group_mem_bytes, stream, #kernel_name,  \
                                   hip_kernel_functor_impl_);                                      \
    } while (0)

#define hipLaunchKernelGGL(kernel_name, num_blocks, dim_blocks, group_mem_bytes, stream, ...)      \
    do {                                                                                           \
        hipLaunchNamedKernelGGL(unnamed, kernel_name, num_blocks, dim_blocks, group_mem_bytes,     \
                                stream, ##__VA_ARGS__);                                            \
    } while (0)

#define hipLaunchKernel(kernel_name, num_blocks, dim_blocks, group_mem_bytes, stream, ...)         \
    do {                                                                                           \
        hipLaunchKernelGGL(kernel_name, num_blocks, dim_blocks, group_mem_bytes, stream,           \
                           hipLaunchParm{}, ##__VA_ARGS__);                                        \
    } while (0)
}  // namespace hip_impl
