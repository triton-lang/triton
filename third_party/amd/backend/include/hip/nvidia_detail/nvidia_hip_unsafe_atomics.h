/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_UNSAFE_ATOMICS_H
#define HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_UNSAFE_ATOMICS_H

__device__ inline float unsafeAtomicAdd(float* addr, float value) {
    return atomicAdd(addr, value);
}

__device__ inline double unsafeAtomicAdd(double* addr, double value) {
#if __CUDA_ARCH__ < 600
    unsigned long long *addr_cast = (unsigned long long*)addr;
    unsigned long long old_val = *addr_cast;
    unsigned long long expected;
    do {
        expected = old_val;
        old_val = atomicCAS(addr_cast, expected,
                            __double_as_longlong(value +
                                                 __longlong_as_double(expected)));
    } while (__double_as_longlong(expected) != __double_as_longlong(old_val));
    return old_val;
#else
    return atomicAdd(addr, value);
#endif
}

__device__ inline float unsafeAtomicMax(float* addr, float value) {
    return atomicMax(addr, value);
}

__device__ inline double unsafeAtomicMax(double* addr, double val) {
    return atomicMax(addr, val);
}

__device__ inline float unsafeAtomicMin(float* addr, float value) {
    return atomicMin(addr, value);
}

__device__ inline double unsafeAtomicMin(double* addr, double val) {
    return atomicMin(addr, val);
}

__device__ inline float safeAtomicAdd(float* addr, float value) {
    return atomicAdd(addr, value);
}

__device__ inline double safeAtomicAdd(double* addr, double value) {
#if __CUDA_ARCH__ < 600
    unsigned long long *addr_cast = (unsigned long long*)addr;
    unsigned long long old_val = *addr_cast;
    unsigned long long expected;
    do {
        expected = old_val;
        old_val = atomicCAS(addr_cast, expected,
                            __double_as_longlong(value +
                                                 __longlong_as_double(expected)));
    } while (__double_as_longlong(expected) != __double_as_longlong(old_val));
    return old_val;
#else
    return atomicAdd(addr, value);
#endif
}

__device__ inline float safeAtomicMax(float* addr, float value) {
    return atomicMax(addr, value);
}

__device__ inline double safeAtomicMax(double* addr, double val) {
    return atomicMax(addr, val);
}

__device__ inline float safeAtomicMin(float* addr, float value) {
    return atomicMin(addr, value);
}

__device__ inline double safeAtomicMin(double* addr, double val) {
    return atomicMin(addr, val);
}

#endif
