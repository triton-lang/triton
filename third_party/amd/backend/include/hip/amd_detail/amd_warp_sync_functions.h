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
// masks. These are enabled by default, and can be disabled by setting the macro
// "HIP_DISABLE_WARP_SYNC_BUILTINS". This arrangement also applies to the
// __activemask() builtin defined in amd_warp_functions.h.
#if !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)

#if !defined(__HIPCC_RTC__)
#include "amd_warp_functions.h"
#include "amd_device_functions.h"
#include "hip_assert.h"
#include <functional>
#include <algorithm>
#endif

extern "C" __device__ __attribute__((const)) int __ockl_wfred_add_i32(int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_add_u32(unsigned int);
extern "C" __device__ __attribute__((const)) int __ockl_wfred_min_i32(int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_min_u32(unsigned int);
extern "C" __device__ __attribute__((const)) int __ockl_wfred_max_i32(int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_max_u32(unsigned int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_and_u32(unsigned int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_or_u32(unsigned int);
extern "C" __device__ __attribute__((const)) unsigned int __ockl_wfred_xor_u32(unsigned int);

#ifdef HIP_ENABLE_EXTRA_WARP_SYNC_TYPES
// this macro enable types that are not in CUDA
extern "C" __device__ __attribute__((const)) long long __ockl_wfred_add_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_add_u64(
    unsigned long long);
extern "C" __device__ __attribute__((const)) float __ockl_wfred_add_f32(float);
extern "C" __device__ __attribute__((const)) double __ockl_wfred_add_f64(double);

extern "C" __device__ __attribute__((const)) long long __ockl_wfred_min_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_min_u64(
    unsigned long long);
extern "C" __device__ __attribute__((const)) float __ockl_wfred_min_f32(float);
extern "C" __device__ __attribute__((const)) double __ockl_wfred_min_f64(double);

extern "C" __device__ __attribute__((const)) long long __ockl_wfred_max_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_max_u64(
    unsigned long long);
extern "C" __device__ __attribute__((const)) float __ockl_wfred_max_f32(float);
extern "C" __device__ __attribute__((const)) double __ockl_wfred_max_f64(double);

extern "C" __device__ __attribute__((const)) int __ockl_wfred_and_i32(int);
extern "C" __device__ __attribute__((const)) long long __ockl_wfred_and_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_and_u64(
    unsigned long long);

extern "C" __device__ __attribute__((const)) int __ockl_wfred_or_i32(int);
extern "C" __device__ __attribute__((const)) long long __ockl_wfred_or_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_or_u64(
    unsigned long long);

extern "C" __device__ __attribute__((const)) int __ockl_wfred_xor_i32(int);
extern "C" __device__ __attribute__((const)) long long __ockl_wfred_xor_i64(long long);
extern "C" __device__ __attribute__((const)) unsigned long long __ockl_wfred_xor_u64(
    unsigned long long);

#endif

template <typename T> __device__ inline T __hip_readfirstlane(T val) {
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
  unsigned long long upper = (unsigned)__builtin_amdgcn_readfirstlane(u.l >> 32);
  u.l = (upper << 32) | lower;
  return u.d;
}

// When compiling for wave32 mode, ignore the upper half of the 64-bit mask.
#define __hip_adjust_mask_for_wave32(MASK)                                                         \
  do {                                                                                             \
    if (static_cast<int>(warpSize) == 32) MASK &= 0xFFFFFFFF;                                      \
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

#define __hip_check_mask(MASK)                                                                     \
  do {                                                                                             \
    __hip_assert(MASK && "mask must be non-zero");                                                 \
    bool done = false;                                                                             \
    while (__any(!done)) {                                                                         \
      if (!done) {                                                                                 \
        auto chosen_mask = __hip_readfirstlane(MASK);                                              \
        if (MASK == chosen_mask) {                                                                 \
          __hip_assert(MASK == __ballot(true) &&                                                   \
                       "all threads specified in the mask"                                         \
                       " must execute the same operation with the same mask");                     \
          done = true;                                                                             \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  } while (0)

#define __hip_do_sync(RETVAL, FUNC, MASK, ...)                                                     \
  do {                                                                                             \
    __hip_assert(MASK && "mask must be non-zero");                                                 \
    bool done = false;                                                                             \
    while (__any(!done)) {                                                                         \
      if (!done) {                                                                                 \
        auto chosen_mask = __hip_readfirstlane(MASK);                                              \
        if (MASK == chosen_mask) {                                                                 \
          __hip_assert(MASK == __ballot(true) &&                                                   \
                       "all threads specified in the mask"                                         \
                       " must execute the same operation with the same mask");                     \
          RETVAL = FUNC(__VA_ARGS__);                                                              \
          done = true;                                                                             \
        }                                                                                          \
      }                                                                                            \
    }                                                                                              \
  } while (0)

__device__ inline void __syncwarp() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

template <typename MaskT> __device__ inline void __syncwarp(MaskT mask) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_check_mask(mask);
  return __syncwarp();
}

// __all_sync, __any_sync, __ballot_sync

template <typename MaskT>
__device__ inline unsigned long long __ballot_sync(MaskT mask, int predicate) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __ballot(predicate) & mask;
}

template <typename MaskT> __device__ inline int __all_sync(MaskT mask, int predicate) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  return __ballot_sync(mask, predicate) == mask;
}

template <typename MaskT> __device__ inline int __any_sync(MaskT mask, int predicate) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  return __ballot_sync(mask, predicate) != 0;
}

// __match_any, __match_all and sync variants

template <typename T> __device__ inline unsigned long long __match_any(T value) {
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
__device__ inline unsigned long long __match_any_sync(MaskT mask, T value) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __match_any(value) & mask;
}

template <typename T> __device__ inline unsigned long long __match_all(T value, int* pred) {
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
__device__ inline unsigned long long __match_all_sync(MaskT mask, T value, int* pred) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  MaskT retval = 0;
  __hip_adjust_mask_for_wave32(mask);
  __hip_do_sync(retval, __match_all, mask, value, pred);
  return retval;
}

// various variants of shfl

template <typename MaskT, typename T>
__device__ inline T __shfl_sync(MaskT mask, T var, int srcLane, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl(var, srcLane, width);
}

template <typename MaskT, typename T>
__device__ inline T __shfl_up_sync(MaskT mask, T var, unsigned int delta, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_up(var, delta, width);
}

template <typename MaskT, typename T>
__device__ inline T __shfl_down_sync(MaskT mask, T var, unsigned int delta, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_down(var, delta, width);
}

template <typename MaskT, typename T>
__device__ inline T __shfl_xor_sync(MaskT mask, T var, int laneMask, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  __hip_check_mask(mask);
  return __shfl_xor(var, laneMask, width);
}

template <typename MaskT, typename T, typename BinaryOp, typename WfReduce>
__device__ inline T __reduce_op_sync(MaskT mask, T val, BinaryOp op, WfReduce wfReduce) {
  using permuteType =
      typename __hip_internal::conditional<sizeof(T) == 4 || sizeof(T) == 2, T, unsigned int>::type;
  static constexpr auto kMaskNumBits = sizeof(MaskT) * 8;
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  unsigned int laneId;
  unsigned int maskIdx;
  // next bit to aggregate with
  int nextBit;

  // if doing the binary reduction tree, this will increase by two in every iteration
  int modulo = 1;
  int leadingZeroes = __clzll(mask);
  int firstLane;
  int lastLane = kMaskNumBits - leadingZeroes - 1;
  int maskNumBits;
  int numIterations;
  // unsigned int[2] is used when T is 64-bit wide
  typename __hip_internal::conditional<sizeof(T) == 4 || sizeof(T) == 2, permuteType,
                                       permuteType[2]>::type result,
      permuteResult;
  auto backwardPermute = [](int index, permuteType val) {
    if constexpr (__hip_internal::is_integral<T>::value ||
                  __hip_internal::is_same<T, double>::value)
      return __hip_ds_bpermute(index, val);
    else
      return __hip_ds_bpermutef(index, val);
  };

  __hip_check_mask(mask);
  maskNumBits = __popcll(mask);

#ifdef __OPTIMIZE__  // at the time of this writing the ockl wfred functions do not compile when
                     // using -O0
  if (maskNumBits == lastLane + 1)
    // this means the mask "does not have holes", and starts from 0; we can use a specific intrinsic
    // to calculate the aggregated result
    return wfReduce(val);
#endif

  firstLane = __builtin_ctzll(mask);
  laneId = __ockl_lane_u32();
  nextBit = laneId;
  // the number of iterations needs to be at least log2(number of bits on)
  numIterations = sizeof(int) * 8 - __clz(maskNumBits);

  if (!(maskNumBits & (maskNumBits - 1)))
    // the number of bits in the mask is a power of 2
    numIterations -= 1;

  maskIdx = __popcll(((1ul << laneId) - 1) & mask);
  mask >>= laneId;
  mask >>= 1ul;

  if constexpr (sizeof(T) == 4 || sizeof(T) == 2)
    result = val;
  else
    __builtin_memcpy(&result, &val, sizeof(T));

  // add the values from the lanes using a reduction tree (first the threads with even-numbered
  // lanes, then multiples of 4, then 8, ...
  while (numIterations) {
    int offset = modulo >> 1;
    int increment = modulo - offset;
    int nextPos = maskIdx + offset + increment;
    bool insideLanes = nextPos < maskNumBits;

    if (insideLanes) {
      int next;

      // find the position to aggregate with; although we could just call fns64() that will probably
      // be very slow when called multiple times in this for loop; this is equivalent
      for (int i = 0; i < increment; i++) {
        next = __builtin_ctzll(mask) + 1;
        mask >>= next;
        nextBit += next;
      }
    }

    if constexpr (sizeof(T) == 2) {
      union {
        int i;
        T f;
      } tmp;

      tmp.f = result;
      tmp.i = __hip_ds_bpermute(nextBit << 2, tmp.i);
      permuteResult = tmp.f;
    } else if constexpr (sizeof(T) == 4)
      permuteResult = backwardPermute(nextBit << 2, result);
    else {
      // ds_bpermute only deals with 32-bit sizes, so for 64-bit types
      // we need to call the permute twice for each half
      permuteResult[0] = backwardPermute(nextBit << 2, result[0]);
      permuteResult[1] = backwardPermute(nextBit << 2, result[1]);
    }

    if (insideLanes) {
      if constexpr (sizeof(T) == 4 || sizeof(T) == 2)
        result = op(result, permuteResult);
      else {
        T tmp;
        unsigned long long rhs =
            (static_cast<unsigned long long>(permuteResult[1]) << 32) | permuteResult[0];

        __builtin_memcpy(&tmp, &result, sizeof(T));
        tmp = op(tmp, *reinterpret_cast<T*>(&rhs));
        __builtin_memcpy(&result, &tmp, sizeof(T));
      }
    }

    modulo <<= 1;
    numIterations--;
  }

  if constexpr (sizeof(T) == 2) {
    union {
      int i;
      T f;
    } tmp;
    tmp.f = result;
    tmp.i = __hip_ds_bpermute(firstLane << 2, tmp.i);
    return tmp.f;
  } else if constexpr (sizeof(T) == 4)
    return backwardPermute(firstLane << 2, result);
  else {
    auto tmp = (static_cast<unsigned long long>(backwardPermute(firstLane << 2, result[1])) << 32) |
               static_cast<unsigned int>(backwardPermute(firstLane << 2, result[0]));
    return *reinterpret_cast<T*>(&tmp);
  }
}

template <typename MaskT> __device__ inline int __reduce_add_sync(MaskT mask, int val) {
  // although C++ has std::plus and other functors, we do not use them because
  // they are in the header <functional> and they were causing problem with hipRTC
  // at this time
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned int __reduce_add_sync(MaskT mask, unsigned int val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline int __reduce_min_sync(MaskT mask, int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned int __reduce_min_sync(MaskT mask, unsigned int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline int __reduce_max_sync(MaskT mask, int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned int __reduce_max_sync(MaskT mask, unsigned int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned int __reduce_or_sync(MaskT mask, unsigned int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs || rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_or_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned int __reduce_and_sync(MaskT mask, unsigned int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs && rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_and_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned int __reduce_xor_sync(MaskT mask, unsigned int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return (!lhs) != (!rhs) == 1; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_xor_u32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

#ifdef HIP_ENABLE_EXTRA_WARP_SYNC_TYPES
template <typename MaskT> __device__ inline long long __reduce_add_sync(MaskT mask, long long val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_add_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline float __reduce_add_sync(MaskT mask, float val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_f32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline double __reduce_add_sync(MaskT mask, double val) {
  auto op = [](decltype(val)& a, decltype(val)& b) { return a + b; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_add_f64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_min_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_min_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline float __reduce_min_sync(MaskT mask, float val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_f32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline double __reduce_min_sync(MaskT mask, double val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return rhs < lhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_min_f64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_max_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_max_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline float __reduce_max_sync(MaskT mask, float val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_f32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline double __reduce_max_sync(MaskT mask, double val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs < rhs ? rhs : lhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_max_f64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline int __reduce_and_sync(MaskT mask, int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs && rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_and_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_and_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs && rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_and_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_and_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs && rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_and_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline int __reduce_or_sync(MaskT mask, int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs || rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_or_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_or_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs || rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_or_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_or_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return lhs || rhs; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_or_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline int __reduce_xor_sync(MaskT mask, int val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return (!lhs) != (!rhs) == 1; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_xor_i32(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT> __device__ inline long long __reduce_xor_sync(MaskT mask, long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return (!lhs) != (!rhs) == 1; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_xor_i64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

template <typename MaskT>
__device__ inline unsigned long long __reduce_xor_sync(MaskT mask, unsigned long long val) {
  auto op = [](decltype(val) lhs, decltype(val) rhs) { return (!lhs) != (!rhs) == 1; };
  auto wfReduce = [](decltype(val) v) { return __ockl_wfred_xor_u64(v); };

  return __reduce_op_sync(mask, val, op, wfReduce);
}

#undef __hip_do_sync
#undef __hip_check_mask
#undef __hip_adjust_mask_for_wave32

#endif  // HIP_ENABLE_EXTRA_WARP_SYNC_TYPES
#endif  // HIP_DISABLE_WARP_SYNC_BUILTINS
