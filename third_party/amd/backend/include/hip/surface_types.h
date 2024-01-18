/*
Copyright (c) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
 *  @file  surface_types.h
 *  @brief Defines surface types for HIP runtime.
 */

#ifndef HIP_INCLUDE_HIP_SURFACE_TYPES_H
#define HIP_INCLUDE_HIP_SURFACE_TYPES_H

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-identifier"
#endif

#if !defined(__HIPCC_RTC__)
#include <hip/driver_types.h>
#endif

/**
 * An opaque value that represents a hip surface object
 */
struct __hip_surface;
typedef struct __hip_surface* hipSurfaceObject_t;

/**
 * hip surface reference
 */
struct surfaceReference {
    hipSurfaceObject_t surfaceObject;
};

/**
 * hip surface boundary modes
 */
enum hipSurfaceBoundaryMode {
    hipBoundaryModeZero = 0,
    hipBoundaryModeTrap = 1,
    hipBoundaryModeClamp = 2
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#endif /* !HIP_INCLUDE_HIP_SURFACE_TYPES_H */
