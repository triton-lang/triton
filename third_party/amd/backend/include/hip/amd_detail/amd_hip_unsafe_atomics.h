/*
Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifdef __cplusplus

/**
 * @brief Unsafe floating point rmw atomic add.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 4 bytes aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and is not supported.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicAdd(float* addr, float value) {
#if defined(__gfx90a__) &&                                                   \
    __has_builtin(__builtin_amdgcn_is_shared) &&                               \
    __has_builtin(__builtin_amdgcn_is_private) &&                              \
    __has_builtin(__builtin_amdgcn_ds_atomic_fadd_f32) &&                      \
    __has_builtin(__builtin_amdgcn_global_atomic_fadd_f32)
  if (__builtin_amdgcn_is_shared(
        (const __attribute__((address_space(0))) void*)addr))
    return __builtin_amdgcn_ds_atomic_fadd_f32(addr, value);
  else if (__builtin_amdgcn_is_private(
              (const __attribute__((address_space(0))) void*)addr)) {
    float temp = *addr;
    *addr = temp + value;
    return temp;
  }
  else
    return __builtin_amdgcn_global_atomic_fadd_f32(addr, value);
#elif __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Unsafe floating point rmw atomic max.
 *
 * Performs a relaxed read-modify-write floating point atomic max with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if greater.
 *
 * @note This operation is currently identical to that performed by
 * atomicMax and is included for completeness.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicMax(float* addr, float val) {
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
}

/**
 * @brief Unsafe floating point rmw atomic min.
 *
 * Performs a relaxed read-modify-write floating point atomic min with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if lesser.
 *
 * @note This operation is currently identical to that performed by
 * atomicMin and is included for completeness.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float unsafeAtomicMin(float* addr, float val) {
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
}

/**
 * @brief Unsafe double precision rmw atomic add.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline double unsafeAtomicAdd(double* addr, double value) {
#if defined(__gfx90a__) && __has_builtin(__builtin_amdgcn_flat_atomic_fadd_f64)
  return __builtin_amdgcn_flat_atomic_fadd_f64(addr, value);
#elif defined (__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Unsafe double precision rmw atomic max.
 *
 * Performs a relaxed read-modify-write double precision atomic max with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if greater.
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double unsafeAtomicMax(double* addr, double val) {
#if (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)) &&  \
    __has_builtin(__builtin_amdgcn_flat_atomic_fmax_f64)
  return __builtin_amdgcn_flat_atomic_fmax_f64(addr, val);
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

/**
 * @brief Unsafe double precision rmw atomic min.
 *
 * Performs a relaxed read-modify-write double precision atomic min with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if lesser.
 *
 * @note This operation currently only performs different operations for
 * the gfx90a target. Other devices continue to use safe atomics.
 *
 * It can be used to generate code that uses fast hardware floating point atomic
 * operations which may handle rounding and subnormal values differently than
 * non-atomic floating point operations.
 *
 * The operation is not always safe and can have undefined behavior unless
 * following condition are met:
 *
 * - \p addr is at least 8 byte aligned
 * - If \p addr is a global segment address, it is in a coarse grain allocation.
 * Passing in global segment addresses in fine grain allocations will result in
 * undefined behavior and are not supported.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double unsafeAtomicMin(double* addr, double val) {
#if (defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)) &&  \
    __has_builtin(__builtin_amdgcn_flat_atomic_fmin_f64)
  return __builtin_amdgcn_flat_atomic_fmin_f64(addr, val);
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

/**
 * @brief Safe floating point rmw atomic add.
 *
 * Performs a relaxed read-modify-write floating point atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicAdd(float* addr, float value) {
#if defined(__gfx908__) || defined(__gfx941__)                                \
    || ((defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx942__))   \
         && !__has_builtin(__hip_atomic_fetch_add))
  // On gfx908, we can generate unsafe FP32 atomic add that does not follow all
  // IEEE rules when -munsafe-fp-atomics is passed. Do a CAS loop emulation instead.
  // On gfx941, we can generate unsafe FP32 atomic add that may not always happen atomically,
  // so we need to force a CAS loop emulation to ensure safety.
  // On gfx90a, gfx940 and gfx942 if we do not have the __hip_atomic_fetch_add builtin, we
  // need to force a CAS loop here.
  float old_val;
#if __has_builtin(__hip_atomic_load)
  old_val = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_load)
  old_val = __uint_as_float(__atomic_load_n(reinterpret_cast<unsigned int*>(addr), __ATOMIC_RELAXED));
#endif // __has_builtin(__hip_atomic_load)
  float expected, temp;
  do {
    temp = expected = old_val;
#if __has_builtin(__hip_atomic_compare_exchange_strong)
    __hip_atomic_compare_exchange_strong(addr, &expected, old_val + value, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_compare_exchange_strong)
    __atomic_compare_exchange_n(addr, &expected, old_val + value, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_compare_exchange_strong)
    old_val = expected;
  } while (__float_as_uint(temp) != __float_as_uint(old_val));
  return old_val;
#elif defined(__gfx90a__)
  // On gfx90a, with the __hip_atomic_fetch_add builtin, relaxed system-scope
  // atomics will produce safe CAS loops, but are otherwise not different than
  // agent-scope atomics. This logic is only applicable for gfx90a, and should
  // not be assumed on other architectures.
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#elif __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Safe floating point rmw atomic max.
 *
 * Performs a relaxed read-modify-write floating point atomic max with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if greater.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicMax(float* addr, float val) {
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
}

/**
 * @brief Safe floating point rmw atomic min.
 *
 * Performs a relaxed read-modify-write floating point atomic min with
 * device memory scope. The original value at \p addr is returned and
 * the value at \p addr is replaced by \p val if lesser.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated
 * @param [in] val Value used to update the value at \p addr.
 * @return Original value contained in \p addr.
 */
__device__ inline float safeAtomicMin(float* addr, float val) {
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
}

/**
 * @brief Safe double precision rmw atomic add.
 *
 * Performs a relaxed read-modify-write double precision atomic add with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated to have the original value plus \p value
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be increment by \p value.
 * @param [in] value Value by \p addr is to be incremented.
 * @return Original value contained in \p addr.
 */
__device__ inline double safeAtomicAdd(double* addr, double value) {
#if defined(__gfx90a__) &&  __has_builtin(__hip_atomic_fetch_add)
  // On gfx90a, with the __hip_atomic_fetch_add builtin, relaxed system-scope
  // atomics will produce safe CAS loops, but are otherwise not different than
  // agent-scope atomics. This logic is only applicable for gfx90a, and should
  // not be assumed on other architectures.
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
#elif defined(__gfx90a__)
  // On gfx90a, if we do not have the __hip_atomic_fetch_add builtin, we need to
  // force a CAS loop here.
  double old_val;
#if __has_builtin(__hip_atomic_load)
  old_val = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_load)
  old_val = __longlong_as_double(__atomic_load_n(reinterpret_cast<unsigned long long*>(addr), __ATOMIC_RELAXED));
#endif // __has_builtin(__hip_atomic_load)
  double expected, temp;
  do {
    temp = expected = old_val;
#if __has_builtin(__hip_atomic_compare_exchange_strong)
    __hip_atomic_compare_exchange_strong(addr, &expected, old_val + value, __ATOMIC_RELAXED,
                                         __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else // !__has_builtin(__hip_atomic_compare_exchange_strong)
    __atomic_compare_exchange_n(addr, &expected, old_val + value, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_compare_exchange_strong)
    old_val = expected;
  } while (__double_as_longlong(temp) != __double_as_longlong(old_val));
  return old_val;
#else // !defined(__gfx90a__)
#if __has_builtin(__hip_atomic_fetch_add)
  return __hip_atomic_fetch_add(addr, value, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else  // !__has_builtin(__hip_atomic_fetch_add)
  return __atomic_fetch_add(addr, value, __ATOMIC_RELAXED);
#endif // __has_builtin(__hip_atomic_fetch_add)
#endif
}

/**
 * @brief Safe double precision rmw atomic max.
 *
 * Performs a relaxed read-modify-write double precision atomic max with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if greater.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double safeAtomicMax(double* addr, double val) {
  #if __has_builtin(__builtin_amdgcn_is_private)
  if (__builtin_amdgcn_is_private(
          (const __attribute__((address_space(0))) void*)addr)) {
    double old = *addr;
    *addr = __builtin_fmax(old, val);
    return old;
  } else {
  #endif
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
  #if __has_builtin(__builtin_amdgcn_is_private)
  }
  #endif
}

/**
 * @brief Safe double precision rmw atomic min.
 *
 * Performs a relaxed read-modify-write double precision atomic min with
 * device memory scope. Original value at \p addr is returned and
 * the value of \p addr is updated with \p val if lesser.
 *
 * @note This operation ensures that, on all targets, we produce safe atomics.
 * This will be the case even when -munsafe-fp-atomics is passed into the compiler.
 *
 * @param [in,out] addr Pointer to value to be updated.
 * @param [in] val Value used to updated the contents at \p addr
 * @return Original value contained at \p addr.
 */
__device__ inline double safeAtomicMin(double* addr, double val) {
  #if __has_builtin(__builtin_amdgcn_is_private)
  if (__builtin_amdgcn_is_private(
           (const __attribute__((address_space(0))) void*)addr)) {
    double old = *addr;
    *addr = __builtin_fmin(old, val);
    return old;
  } else {
  #endif
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
  #if __has_builtin(__builtin_amdgcn_is_private)
  }
  #endif
}

#endif
