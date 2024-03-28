/*
Copyright (c) 2015 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

/**
 *  @file  amd_detail/hip_cooperative_groups_helper.h
 *
 *  @brief Device side implementation of cooperative group feature.
 *
 *  Defines helper constructs and APIs which aid the types and device API
 *  wrappers defined within `amd_detail/hip_cooperative_groups.h`.
 */
#ifndef HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
#define HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H

#if __cplusplus
#if !defined(__HIPCC_RTC__)
#include <hip/amd_detail/amd_hip_runtime.h> // threadId, blockId
#include <hip/amd_detail/amd_device_functions.h>
#endif
#if !defined(__align__)
#define __align__(x) __attribute__((aligned(x)))
#endif

#if !defined(__CG_QUALIFIER__)
#define __CG_QUALIFIER__ __device__ __forceinline__
#endif

#if !defined(__CG_STATIC_QUALIFIER__)
#define __CG_STATIC_QUALIFIER__ __device__ static __forceinline__
#endif

#if !defined(_CG_STATIC_CONST_DECL_)
#define _CG_STATIC_CONST_DECL_ static constexpr
#endif

#if __AMDGCN_WAVEFRONT_SIZE == 32
using lane_mask = unsigned int;
#else
using lane_mask = unsigned long long int;
#endif

namespace cooperative_groups {

/* Global scope */
template <unsigned int size>
using is_power_of_2 = std::integral_constant<bool, (size & (size - 1)) == 0>;

template <unsigned int size>
using is_valid_wavefront = std::integral_constant<bool, (size <= __AMDGCN_WAVEFRONT_SIZE)>;

template <unsigned int size>
using is_valid_tile_size =
    std::integral_constant<bool, is_power_of_2<size>::value && is_valid_wavefront<size>::value>;

template <typename T>
using is_valid_type =
    std::integral_constant<bool, std::is_integral<T>::value || std::is_floating_point<T>::value>;

namespace internal {

/**
* @brief Enums representing different cooperative group types
* @note  This enum is only applicable on Linux.
*
 */
typedef enum {
  cg_invalid,
  cg_multi_grid,
  cg_grid,
  cg_workgroup,
  cg_tiled_group,
  cg_coalesced_group
} group_type;
/**
 *  @ingroup CooperativeG
 *  @{
 *  This section describes the cooperative groups functions of HIP runtime API.
 *  
 *  The cooperative groups provides flexible thread parallel programming algorithms, threads
 *  cooperate and share data to perform collective computations.
 *
 *  @note  Cooperative groups feature is implemented on Linux, under developement
 *  on Windows.
 *
 */
/**
 *
 * @brief  Functionalities related to multi-grid cooperative group type
 * @note  The following cooperative groups functions are only applicable on Linux.
 *
 */
namespace multi_grid {

__CG_STATIC_QUALIFIER__ uint32_t num_grids() {
  return static_cast<uint32_t>(__ockl_multi_grid_num_grids()); }

__CG_STATIC_QUALIFIER__ uint32_t grid_rank() {
  return static_cast<uint32_t>(__ockl_multi_grid_grid_rank()); }

__CG_STATIC_QUALIFIER__ uint32_t size() { return static_cast<uint32_t>(__ockl_multi_grid_size()); }

__CG_STATIC_QUALIFIER__ uint32_t thread_rank() {
  return static_cast<uint32_t>(__ockl_multi_grid_thread_rank()); }

__CG_STATIC_QUALIFIER__ bool is_valid() { return static_cast<bool>(__ockl_multi_grid_is_valid()); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_multi_grid_sync(); }

}  // namespace multi_grid

/**
 *  @brief Functionalities related to grid cooperative group type
 *  @note  The following cooperative groups functions are only applicable on Linux.
 */
namespace grid {

__CG_STATIC_QUALIFIER__ uint32_t size() {
  return static_cast<uint32_t>((blockDim.z * gridDim.z) * (blockDim.y * gridDim.y) *
                    (blockDim.x * gridDim.x));
}

__CG_STATIC_QUALIFIER__ uint32_t thread_rank() {
  // Compute global id of the workgroup to which the current thread belongs to
  uint32_t blkIdx = static_cast<uint32_t>((blockIdx.z * gridDim.y * gridDim.x) +
                               (blockIdx.y * gridDim.x) + (blockIdx.x));

  // Compute total number of threads being passed to reach current workgroup
  // within grid
  uint32_t num_threads_till_current_workgroup =
      static_cast<uint32_t>(blkIdx * (blockDim.x * blockDim.y * blockDim.z));

  // Compute thread local rank within current workgroup
  uint32_t local_thread_rank = static_cast<uint32_t>((threadIdx.z * blockDim.y * blockDim.x) +
                                          (threadIdx.y * blockDim.x) + (threadIdx.x));

  return (num_threads_till_current_workgroup + local_thread_rank);
}

__CG_STATIC_QUALIFIER__ bool is_valid() { return static_cast<bool>(__ockl_grid_is_valid()); }

__CG_STATIC_QUALIFIER__ void sync() { __ockl_grid_sync(); }

}  // namespace grid

/**
 *  @brief Functionalities related to `workgroup` (thread_block in CUDA terminology)
 *  cooperative group type
 *  @note  The following cooperative groups functions are only applicable on Linux.
 */
namespace workgroup {

__CG_STATIC_QUALIFIER__ dim3 group_index() {
  return (dim3(static_cast<uint32_t>(blockIdx.x), static_cast<uint32_t>(blockIdx.y),
               static_cast<uint32_t>(blockIdx.z)));
}

__CG_STATIC_QUALIFIER__ dim3 thread_index() {
  return (dim3(static_cast<uint32_t>(threadIdx.x), static_cast<uint32_t>(threadIdx.y),
               static_cast<uint32_t>(threadIdx.z)));
}

__CG_STATIC_QUALIFIER__ uint32_t size() {
  return (static_cast<uint32_t>(blockDim.x * blockDim.y * blockDim.z));
}

__CG_STATIC_QUALIFIER__ uint32_t thread_rank() {
  return (static_cast<uint32_t>((threadIdx.z * blockDim.y * blockDim.x) +
                     (threadIdx.y * blockDim.x) + (threadIdx.x)));
}

__CG_STATIC_QUALIFIER__ bool is_valid() {
  return true;
}

__CG_STATIC_QUALIFIER__ void sync() { __syncthreads(); }

__CG_STATIC_QUALIFIER__ dim3 block_dim() {
  return (dim3(static_cast<uint32_t>(blockDim.x), static_cast<uint32_t>(blockDim.y),
          static_cast<uint32_t>(blockDim.z)));
}

}  // namespace workgroup

namespace tiled_group {

// enforce ordering for memory intructions
__CG_STATIC_QUALIFIER__ void sync() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent"); }

}  // namespace tiled_group

namespace coalesced_group {

// enforce ordering for memory intructions
__CG_STATIC_QUALIFIER__ void sync() { __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "agent"); }

// Masked bit count
//
// For each thread, this function returns the number of active threads which
// have i-th bit of x set and come before the current thread.
__CG_STATIC_QUALIFIER__ unsigned int masked_bit_count(lane_mask x, unsigned int add = 0) {
  unsigned int counter=0;
    #if __AMDGCN_WAVEFRONT_SIZE == 32
      counter = __builtin_amdgcn_mbcnt_lo(x, add);
    #else
      counter = __builtin_amdgcn_mbcnt_lo(static_cast<lane_mask>(x), add);
      counter = __builtin_amdgcn_mbcnt_hi(static_cast<lane_mask>(x >> 32), counter);
    #endif

    return counter;
}

}  // namespace coalesced_group


}  // namespace internal

}  // namespace cooperative_groups
/**
*  @}
*/

#endif  // __cplusplus
#endif  // HIP_INCLUDE_HIP_AMD_DETAIL_HIP_COOPERATIVE_GROUPS_HELPER_H
