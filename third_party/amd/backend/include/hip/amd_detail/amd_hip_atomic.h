/*
Copyright (c) 2015 - Present Advanced Micro Devices, Inc. All rights reserved.

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

#if !defined(__HIPCC_RTC__)
#include "amd_device_functions.h"
#endif

#if __has_builtin(__hip_atomic_compare_exchange_strong)

template<bool B, typename T, typename F> struct Cond_t;

template<typename T, typename F> struct Cond_t<true, T, F> { using type = T; };
template<typename T, typename F> struct Cond_t<false, T, F> { using type = F; };

#if !__HIP_DEVICE_COMPILE__
//TODO: Remove this after compiler pre-defines the following Macros.
#define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#define __HIP_MEMORY_SCOPE_WORKGROUP 3
#define __HIP_MEMORY_SCOPE_AGENT 4
#define __HIP_MEMORY_SCOPE_SYSTEM 5
#endif

#if !defined(__HIPCC_RTC__)
#include "amd_hip_unsafe_atomics.h"
#endif

// Atomic expanders
template<
  int mem_order = __ATOMIC_SEQ_CST,
  int mem_scope= __HIP_MEMORY_SCOPE_SYSTEM,
  typename T,
  typename Op,
  typename F>
inline
__attribute__((always_inline, device))
T hip_cas_expander(T* p, T x, Op op, F f) noexcept
{
  using FP = __attribute__((address_space(0))) const void*;

  __device__
  extern bool is_shared_workaround(FP) asm("llvm.amdgcn.is.shared");

  if (is_shared_workaround((FP)p))
    return f();

  using U = typename Cond_t<
    sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>::type;

  auto q = reinterpret_cast<U*>(p);

  U tmp0{__hip_atomic_load(q, mem_order, mem_scope)};
  U tmp1;
  do {
    tmp1 = tmp0;

    op(reinterpret_cast<T&>(tmp1), x);
  } while (!__hip_atomic_compare_exchange_strong(q, &tmp0, tmp1, mem_order,
                                                 mem_order, mem_scope));

  return reinterpret_cast<const T&>(tmp0);
}

template<
  int mem_order = __ATOMIC_SEQ_CST,
  int mem_scope= __HIP_MEMORY_SCOPE_SYSTEM,
  typename T,
  typename Cmp,
  typename F>
inline
__attribute__((always_inline, device))
T hip_cas_extrema_expander(T* p, T x, Cmp cmp, F f) noexcept
{
  using FP = __attribute__((address_space(0))) const void*;

  __device__
  extern bool is_shared_workaround(FP) asm("llvm.amdgcn.is.shared");

  if (is_shared_workaround((FP)p))
    return f();

  using U = typename Cond_t<
    sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>::type;

  auto q = reinterpret_cast<U*>(p);

  U tmp{__hip_atomic_load(q, mem_order, mem_scope)};
  while (cmp(x, reinterpret_cast<const T&>(tmp)) &&
         !__hip_atomic_compare_exchange_strong(q, &tmp, x, mem_order, mem_order,
                                               mem_scope));

  return reinterpret_cast<const T&>(tmp);
}

__device__
inline
int atomicCAS(int* address, int compare, int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
    return compare;
}

__device__
inline
int atomicCAS_system(int* address, int compare, int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
    return compare;
}

__device__
inline
unsigned int atomicCAS(unsigned int* address, unsigned int compare, unsigned int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__
inline
unsigned int atomicCAS_system(unsigned int* address, unsigned int compare, unsigned int val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return compare;
}

__device__
inline
unsigned long atomicCAS(unsigned long* address, unsigned long compare, unsigned long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__
inline
unsigned long atomicCAS_system(unsigned long* address, unsigned long compare, unsigned long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return compare;
}

__device__
inline
unsigned long long atomicCAS(unsigned long long* address, unsigned long long compare,
                             unsigned long long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
  return compare;
}

__device__
inline
unsigned long long atomicCAS_system(unsigned long long* address, unsigned long long compare,
                                    unsigned long long val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
  return compare;
}

__device__
inline
float atomicCAS(float* address, float compare, float val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
    return compare;
}

__device__
inline
float atomicCAS_system(float* address, float compare, float val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
    return compare;
}

__device__
inline
double atomicCAS(double* address, double compare, double val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_AGENT);
    return compare;
}

__device__
inline
double atomicCAS_system(double* address, double compare, double val) {
  __hip_atomic_compare_exchange_strong(address, &compare, val, __ATOMIC_RELAXED, __ATOMIC_RELAXED,
                                       __HIP_MEMORY_SCOPE_SYSTEM);
    return compare;
}

__device__
inline
int atomicAdd(int* address, int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicAdd_system(int* address, int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicAdd(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicAdd_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long atomicAdd(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long atomicAdd_system(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicAdd(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicAdd_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
float atomicAdd(float* address, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, val);
#else
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__
inline
float atomicAdd_system(float* address, float val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

#if !defined(__HIPCC_RTC__)
DEPRECATED("use atomicAdd instead")
#endif // !defined(__HIPCC_RTC__)
__device__
inline
void atomicAddNoRet(float* address, float val)
{
    __ockl_atomic_add_noret_f32(address, val);
}

__device__
inline
double atomicAdd(double* address, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, val);
#else
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__
inline
double atomicAdd_system(double* address, double val) {
  return __hip_atomic_fetch_add(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicSub(int* address, int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicSub_system(int* address, int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicSub(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicSub_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long atomicSub(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long atomicSub_system(unsigned long* address, unsigned long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicSub(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicSub_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
float atomicSub(float* address, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, -val);
#else
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__
inline
float atomicSub_system(float* address, float val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
double atomicSub(double* address, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicAdd(address, -val);
#else
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif
}

__device__
inline
double atomicSub_system(double* address, double val) {
  return __hip_atomic_fetch_add(address, -val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicExch(int* address, int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
int atomicExch_system(int* address, int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned int atomicExch(unsigned int* address, unsigned int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned int atomicExch_system(unsigned int* address, unsigned int val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long atomicExch(unsigned long* address, unsigned long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long atomicExch_system(unsigned long* address, unsigned long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
unsigned long long atomicExch(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
unsigned long long atomicExch_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
float atomicExch(float* address, float val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
float atomicExch_system(float* address, float val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
double atomicExch(double* address, double val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__
inline
double atomicExch_system(double* address, double val) {
  return __hip_atomic_exchange(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__device__
inline
int atomicMin(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int x, int y) { return x < y; }, [=]() {
      return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicMin_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int x, int y) { return x < y; }, [=]() {
      return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicMin(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int x, unsigned int y) { return x < y; }, [=]() {
      return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__

}

__device__
inline
unsigned int atomicMin_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int x, unsigned int y) { return x < y; }, [=]() {
      return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMin(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long x, unsigned long y) { return x < y; },
    [=]() {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicMin_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long x, unsigned long y) { return x < y; },
    [=]() {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMin(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long x, unsigned long long y) { return x < y; },
    [=]() {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMin_system(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long long x, unsigned long long y) { return x < y; },
    [=]() {
    return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
long long atomicMin(long long* address, long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
      address, val, [](long long x, long long y) { return x < y; },
      [=]() {
        return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif  // __gfx941__
}

__device__
inline
long long atomicMin_system(long long* address, long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
      address, val, [](long long x, long long y) { return x < y; },
      [=]() {
        return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      });
#else
  return __hip_atomic_fetch_min(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif  // __gfx941__
}

__device__
inline
float atomicMin(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value > val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned int *uaddr = (unsigned int *)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
  #endif
#endif
}

__device__
inline
float atomicMin_system(float* address, float val) {
  unsigned int* uaddr { reinterpret_cast<unsigned int*>(address) };
  #if __has_builtin(__hip_atomic_load)
    unsigned int tmp {__hip_atomic_load(uaddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
  #else
    unsigned int tmp {__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
  #endif
  float value = __uint_as_float(tmp);

  while (val < value) {
    value = atomicCAS_system(address, value, val);
  }

  return value;
}

__device__
inline
double atomicMin(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMin(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value > val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) > val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
  #endif
#endif
}

__device__
inline
double atomicMin_system(double* address, double val) {
  unsigned long long* uaddr { reinterpret_cast<unsigned long long*>(address) };
  #if __has_builtin(__hip_atomic_load)
    unsigned long long tmp {__hip_atomic_load(uaddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
  #else
    unsigned long long tmp {__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
  #endif
  double value = __longlong_as_double(tmp);

  while (val < value) {
    value = atomicCAS_system(address, value, val);
  }

  return value;
}

__device__
inline
int atomicMax(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int x, int y) { return y < x; }, [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicMax_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int x, int y) { return y < x; }, [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicMax(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int x, unsigned int y) { return y < x; }, [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicMax_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int x, unsigned int y) { return y < x; }, [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicMax(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long x, unsigned long y) { return y < x; },
    [=]() {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicMax_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long x, unsigned long y) { return y < x; },
    [=]() {
    return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMax(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long x, unsigned long long y) { return y < x; },
    [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicMax_system(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long long x, unsigned long long y) { return y < x; },
    [=]() {
      return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED,
                                    __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
long long atomicMax(long long* address, long long val) {
  #if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
      address, val, [](long long x, long long y) { return y < x; },
      [=]() {
        return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
long long atomicMax_system(long long* address, long long val) {
#if defined(__gfx941__)
  return hip_cas_extrema_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
      address, val, [](long long x, long long y) { return y < x; },
      [=]() {
        return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      });
#else
  return __hip_atomic_fetch_max(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif  // __gfx941__
}

__device__
inline
float atomicMax(float* addr, float val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  float value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value < val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned int *uaddr = (unsigned int *)addr;
  unsigned int value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __uint_as_float(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __float_as_uint(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __uint_as_float(value);
  #endif
#endif
}

__device__
inline
float atomicMax_system(float* address, float val) {
  unsigned int* uaddr { reinterpret_cast<unsigned int*>(address) };
  #if __has_builtin(__hip_atomic_load)
    unsigned int tmp {__hip_atomic_load(uaddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
  #else
    unsigned int tmp {__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
  #endif
  float value = __uint_as_float(tmp);

  while (value < val) {
    value = atomicCAS_system(address, value, val);
  }

  return value;
}

__device__
inline
double atomicMax(double* addr, double val) {
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
  return unsafeAtomicMax(addr, val);
#else
  #if __has_builtin(__hip_atomic_load) && \
      __has_builtin(__hip_atomic_compare_exchange_strong)
  double value = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  bool done = false;
  while (!done && value < val) {
    done = __hip_atomic_compare_exchange_strong(addr, &value, val,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  return value;
  #else
  unsigned long long *uaddr = (unsigned long long *)addr;
  unsigned long long value = __atomic_load_n(uaddr, __ATOMIC_RELAXED);
  bool done = false;
  while (!done && __longlong_as_double(value) < val) {
    done = __atomic_compare_exchange_n(uaddr, &value, __double_as_longlong(val), false,
               __ATOMIC_RELAXED, __ATOMIC_RELAXED);
  }
  return __longlong_as_double(value);
  #endif
#endif
}

__device__
inline
double atomicMax_system(double* address, double val) {
  unsigned long long* uaddr { reinterpret_cast<unsigned long long*>(address) };
  #if __has_builtin(__hip_atomic_load)
    unsigned long long tmp {__hip_atomic_load(uaddr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM)};
  #else
    unsigned long long tmp {__atomic_load_n(uaddr, __ATOMIC_RELAXED)};
  #endif
  double value = __longlong_as_double(tmp);

  while (value < val) {
      value = atomicCAS_system(address, value, val);
  }

  return value;
}

__device__
inline
unsigned int atomicInc(unsigned int* address, unsigned int val)
{
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned int& x, unsigned int y) { x = (x >= y) ? 0 : (x + 1); },
    [=]() {
    return
      __builtin_amdgcn_atomic_inc32(address, val, __ATOMIC_RELAXED, "agent");
  });
#else
    return __builtin_amdgcn_atomic_inc32(address, val, __ATOMIC_RELAXED, "agent");
#endif // __gfx941__

}

__device__
inline
unsigned int atomicDec(unsigned int* address, unsigned int val)
{
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned int& x, unsigned int y) { x = (!x || x > y) ? y : (x - 1); },
    [=]() {
    return
      __builtin_amdgcn_atomic_dec32(address, val, __ATOMIC_RELAXED, "agent");
  });
#else
  return __builtin_amdgcn_atomic_dec32(address, val, __ATOMIC_RELAXED, "agent");
#endif // __gfx941__

}

__device__
inline
int atomicAnd(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int& x, int y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicAnd_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int& x, int y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicAnd(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int& x, unsigned int y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicAnd_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int& x, unsigned int y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicAnd(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned long& x, unsigned long y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicAnd_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned long& x, unsigned long y) { x &= y; }, [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicAnd(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x &= y; },
    [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicAnd_system(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x &= y; },
    [=]() {
    return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_and(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
int atomicOr(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int& x, int y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicOr_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int& x, int y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicOr(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int& x, unsigned int y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicOr_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int& x, unsigned int y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicOr(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned long& x, unsigned long y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicOr_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned long& x, unsigned long y) { x |= y; }, [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicOr(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x |= y; },
    [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicOr_system(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x |= y; },
    [=]() {
    return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_or(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
int atomicXor(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](int& x, int y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
int atomicXor_system(int* address, int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](int& x, int y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicXor(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned int& x, unsigned int y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned int atomicXor_system(unsigned int* address, unsigned int val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned int& x, unsigned int y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicXor(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address, val, [](unsigned long& x, unsigned long y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long atomicXor_system(unsigned long* address, unsigned long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM>(
    address, val, [](unsigned long& x, unsigned long y) { x ^= y; }, [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_SYSTEM);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicXor(unsigned long long* address, unsigned long long val) {
#if defined(__gfx941__)
  return hip_cas_expander<__ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT>(
    address,
    val,
    [](unsigned long long& x, unsigned long long y) { x ^= y; },
    [=]() {
    return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED,
                                  __HIP_MEMORY_SCOPE_AGENT);
  });
#else
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#endif // __gfx941__
}

__device__
inline
unsigned long long atomicXor_system(unsigned long long* address, unsigned long long val) {
  return __hip_atomic_fetch_xor(address, val, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

#else // __hip_atomic_compare_exchange_strong

__device__
inline
int atomicCAS(int* address, int compare, int val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}
__device__
inline
unsigned int atomicCAS(
    unsigned int* address, unsigned int compare, unsigned int val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}
__device__
inline
unsigned long long atomicCAS(
    unsigned long long* address,
    unsigned long long compare,
    unsigned long long val)
{
    __atomic_compare_exchange_n(
        address, &compare, val, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);

    return compare;
}

__device__
inline
int atomicAdd(int* address, int val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicAdd(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicAdd(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
}
__device__
inline
float atomicAdd(float* address, float val)
{
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
    return unsafeAtomicAdd(address, val);
#else
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#endif
}

#if !defined(__HIPCC_RTC__)
DEPRECATED("use atomicAdd instead")
#endif // !defined(__HIPCC_RTC__)
__device__
inline
void atomicAddNoRet(float* address, float val)
{
    __ockl_atomic_add_noret_f32(address, val);
}

__device__
inline
double atomicAdd(double* address, double val)
{
#if defined(__AMDGCN_UNSAFE_FP_ATOMICS__)
    return unsafeAtomicAdd(address, val);
#else
    return __atomic_fetch_add(address, val, __ATOMIC_RELAXED);
#endif
}

__device__
inline
int atomicSub(int* address, int val)
{
    return __atomic_fetch_sub(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicSub(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_sub(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicExch(int* address, int val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicExch(unsigned int* address, unsigned int val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicExch(unsigned long long* address, unsigned long long val)
{
    return __atomic_exchange_n(address, val, __ATOMIC_RELAXED);
}
__device__
inline
float atomicExch(float* address, float val)
{
    return __uint_as_float(__atomic_exchange_n(
        reinterpret_cast<unsigned int*>(address),
        __float_as_uint(val),
        __ATOMIC_RELAXED));
}

__device__
inline
int atomicMin(int* address, int val)
{
    return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicMin(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_min(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicMin(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (val < tmp) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) { tmp = tmp1; continue; }

        tmp = atomicCAS(address, tmp, val);
    }

    return tmp;
}
__device__ inline long long atomicMin(long long* address, long long val) {
    long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (val < tmp) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) {
          tmp = tmp1;
          continue;
        }

        tmp = atomicCAS(address, tmp, val);
    }
    return tmp;
}

__device__
inline
int atomicMax(int* address, int val)
{
    return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicMax(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_max(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicMax(
    unsigned long long* address, unsigned long long val)
{
    unsigned long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (tmp < val) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) { tmp = tmp1; continue; }

        tmp = atomicCAS(address, tmp, val);
    }

    return tmp;
}
__device__ inline long long atomicMax(long long* address, long long val) {
    long long tmp{__atomic_load_n(address, __ATOMIC_RELAXED)};
    while (tmp < val) {
        const auto tmp1 = __atomic_load_n(address, __ATOMIC_RELAXED);

        if (tmp1 != tmp) {
          tmp = tmp1;
          continue;
        }

        tmp = atomicCAS(address, tmp, val);
    }
    return tmp;
}

__device__
inline
unsigned int atomicInc(unsigned int* address, unsigned int val)
{
  return __builtin_amdgcn_atomic_inc32(address, val, __ATOMIC_RELAXED, "agent");
}

__device__
inline
unsigned int atomicDec(unsigned int* address, unsigned int val)
{
  return __builtin_amdgcn_atomic_dec32(address, val, __ATOMIC_RELAXED, "agent");
}

__device__
inline
int atomicAnd(int* address, int val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicAnd(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicAnd(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_and(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicOr(int* address, int val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicOr(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicOr(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_or(address, val, __ATOMIC_RELAXED);
}

__device__
inline
int atomicXor(int* address, int val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned int atomicXor(unsigned int* address, unsigned int val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}
__device__
inline
unsigned long long atomicXor(
    unsigned long long* address, unsigned long long val)
{
    return __atomic_fetch_xor(address, val, __ATOMIC_RELAXED);
}

#endif // __hip_atomic_compare_exchange_strong
