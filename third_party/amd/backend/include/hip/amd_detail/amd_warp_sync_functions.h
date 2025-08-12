/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

// Warp sync builtins (with explicit mask argument) introduced in ROCm 6.2 as a
// preview to allow end-users to adapt to the new interface involving 64-bit
// masks. These are disabled by default, and can be enabled by setting the macro
// below. The builtins will be enabled unconditionally in ROCm 6.3.
//
// This arrangement also applies to the __activemask() builtin defined in
// amd_warp_functions.h.
#ifdef HIP_ENABLE_WARP_SYNC_BUILTINS

#if !defined(__HIPCC_RTC__)
#include "amd_warp_functions.h"
#include "hip_assert.h"
#endif

template <typename T>
__device__ inline
T __hip_readfirstlane(T val) {
  // In theory, behaviour is undefined when reading from a union member other
  // than the member that was last assigned to, but it works in practice because
  // we rely on the compiler to do the reasonable thing.
  union {
    unsigned long long l;
    T d;
  } u;
  u.d = val;
  // NOTE: The builtin returns int, so we first cast it to unsigned int and only
  // then extend it to 64 bits.
  unsigned long long lower = (unsigned)__builtin_amdgcn_readfirstlane(u.l);
  unsigned long long upper =
      (unsigned)__builtin_amdgcn_readfirstlane(u.l >> 32);
  u.l = (upper << 32) | lower;
  return u.d;
}

// When compiling for wave32 mode, ignore the upper half of the 64-bit mask.
#define __hip_adjust_mask_for_wave32(MASK)            \
  do {                                          \
    if (warpSize == 32) MASK &= 0xFFFFFFFF;     \
  } while (0)

// We use a macro to expand each builtin into a waterfall that implements the
// mask semantics:
//
// 1. The mask argument may be divergent.
// 2. Each active thread must have its own bit set in its own mask value.
// 3. For a given mask value, all threads that are mentioned in the mask must
//    execute the same static instance of the builtin with the same mask.
// 4. The union of all mask values supplied at a static instance must be equal
//    to the activemask at the program point.
//
// Thus, the mask argument partitions the set of currently active threads in the
// wave into disjoint subsets that cover all active threads.
//
// Implementation notes:
// ---------------------
//
// We implement this as a waterfall loop that executes the builtin for each
// subset separately. The return value is a divergent value across the active
// threads. The value for inactive threads is defined by each builtin
// separately.
//
// As long as every mask value is non-zero, we don't need to check if a lane
// specifies itself in the mask; that is done by the later assertion where all
// chosen lanes must be in the chosen mask.

#define __hip_check_mask(MASK)                                                 \
  do {                                                                         \
    __hip_assert(MASK && "mask must be non-zero");                             \
    bool done = false;                                                         \
    while (__any(!done)) {                                                     \
      if (!done) {                                                             \
        auto chosen_mask = __hip_readfirstlane(MASK);                          \
        if (MASK == chosen_mask) {                                             \
          __hip_assert(MASK == __ballot(true) &&                               \
                       "all threads specified in the mask"                     \
                       " must execute the same operation with the same mask"); \
          done = true;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while(0)

#define __hip_do_sync(RETVAL, FUNC, MASK, ...)                                 \
  do {                                                                         \
    __hip_assert(MASK && "mask must be non-zero");                             \
    bool done = false;                                                         \
    while (__any(!done)) {                                                     \
      if (!done) {                                                             \
        auto chosen_mask = __hip_readfirstlane(MASK);                          \
        if (MASK == chosen_mask) {                                             \
          __hip_assert(MASK == __ballot(true) &&                               \
                       "all threads specified in the mask"                     \
                       " must execute the same operation with the same mask"); \
          RETVAL = FUNC(__VA_ARGS__);                                          \
          done = true;                                                         \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while(0)

// __all_sync, __any_sync, __ballot_sync

template <typename MaskT>
__device__ inline
unsigned long long __ballot_sync(MaskT mask, int predicate) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __ballot(predicate) & mask;
}

template <typename MaskT>
__device__ inline
int __all_sync(MaskT mask, int predicate) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  return __ballot_sync(mask, predicate) == mask;
}

template <typename MaskT>
__device__ inline
int __any_sync(MaskT mask, int predicate) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  return __ballot_sync(mask, predicate) != 0;
}

// __match_any, __match_all and sync variants

template <typename T>
__device__ inline
unsigned long long __match_any(T value) {
  static_assert(
      (__hip_internal::is_integral<T>::value || __hip_internal::is_floating_point<T>::value) &&
          (sizeof(T) == 4 || sizeof(T) == 8),
      "T can be int, unsigned int, long, unsigned long, long long, unsigned "
      "long long, float or double.");
  bool done = false;
  unsigned long long retval = 0;

  while (__any(!done)) {
    if (!done) {
      T chosen = __hip_readfirstlane(value);
      if (chosen == value) {
        retval = __activemask();
        done = true;
      }
    }
  }

  return retval;
}

template <typename MaskT, typename T>
__device__ inline
unsigned long long __match_any_sync(MaskT mask, T value) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __match_any(value) & mask;
}

template <typename T>
__device__ inline
unsigned long long __match_all(T value, int* pred) {
  static_assert(
      (__hip_internal::is_integral<T>::value || __hip_internal::is_floating_point<T>::value) &&
          (sizeof(T) == 4 || sizeof(T) == 8),
      "T can be int, unsigned int, long, unsigned long, long long, unsigned "
      "long long, float or double.");
  T first = __hip_readfirstlane(value);
  if (__all(first == value)) {
    *pred = true;
    return __activemask();
  } else {
    *pred = false;
    return 0;
  }
}

template <typename MaskT, typename T>
__device__ inline
unsigned long long __match_all_sync(MaskT mask, T value, int* pred) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  MaskT retval = 0;
  __hip_adjust_mask_for_wave32(mask);
  __hip_do_sync(retval, __match_all, mask, value, pred);
  return retval;
}

// various variants of shfl

template <typename MaskT, typename T>
__device__ inline
T __shfl_sync(MaskT mask, T var, int srcLane,
              int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl(var, srcLane, width);
}

template <typename MaskT, typename T>
__device__ inline
T __shfl_up_sync(MaskT mask, T var, unsigned int delta,
                                   int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_up(var, delta, width);
}

template <typename MaskT, typename T>
__device__ inline
T __shfl_down_sync(MaskT mask, T var, unsigned int delta,
                                     int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_down(var, delta, width);
}

template <typename MaskT, typename T>
__device__ inline
T __shfl_xor_sync(MaskT mask, T var, int laneMask,
                                    int width = __AMDGCN_WAVEFRONT_SIZE) {
  static_assert(
      __hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
      "The mask must be a 64-bit integer. "
      "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_xor(var, laneMask, width);
}

#undef __hip_do_sync
#undef __hip_check_mask
#undef __hip_adjust_mask_for_wave32

#endif // HIP_ENABLE_WARP_SYNC_BUILTINS
