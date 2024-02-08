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

#ifndef HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_ATOMICS_H
#define HIP_INCLUDE_HIP_NVIDIA_DETAIL_HIP_ATOMICS_H


__device__ inline float atomicMax(float* addr, float val) {
    int ret = __float_as_int(*addr);
    while (val > __int_as_float(ret)) {
        int old = ret;
        if ((ret = atomicCAS((int *)addr, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}
__device__ inline double atomicMax(double* addr, double val) {
    unsigned long long ret = __double_as_longlong(*addr);
    while (val > __longlong_as_double(ret)) {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *)addr, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

__device__ inline float atomicMin(float* addr, float val) {
    int ret = __float_as_int(*addr);
    while (val < __int_as_float(ret)) {
        int old = ret;
        if ((ret = atomicCAS((int *)addr, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ inline double atomicMin(double* addr, double val) {
    unsigned long long ret = __double_as_longlong(*addr);
    while (val < __longlong_as_double(ret)) {
        unsigned long long old = ret;
        if ((ret = atomicCAS((unsigned long long *)addr, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}


#endif
