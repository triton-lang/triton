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

#ifndef HIP_INCLUDE_HIP_LIBRARY_TYPES_H
#define HIP_INCLUDE_HIP_LIBRARY_TYPES_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#endif

#if defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)

typedef enum hipDataType {
  HIP_R_32F   =  0,
  HIP_R_64F   =  1,
  HIP_R_16F   =  2,
  HIP_R_8I    =  3,
  HIP_C_32F   =  4,
  HIP_C_64F   =  5,
  HIP_C_16F   =  6,
  HIP_C_8I    =  7,
  HIP_R_8U    =  8,
  HIP_C_8U    =  9,
  HIP_R_32I   = 10,
  HIP_C_32I   = 11,
  HIP_R_32U   = 12,
  HIP_C_32U   = 13,
  HIP_R_16BF  = 14,
  HIP_C_16BF  = 15,
  HIP_R_4I    = 16,
  HIP_C_4I    = 17,
  HIP_R_4U    = 18,
  HIP_C_4U    = 19,
  HIP_R_16I   = 20,
  HIP_C_16I   = 21,
  HIP_R_16U   = 22,
  HIP_C_16U   = 23,
  HIP_R_64I   = 24,
  HIP_C_64I   = 25,
  HIP_R_64U   = 26,
  HIP_C_64U   = 27,
  // HIP specific Data Types
  HIP_R_8F_E4M3_FNUZ = 1000,
  HIP_R_8F_E5M2_FNUZ = 1001
} hipDataType;

typedef enum hipLibraryPropertyType {
  HIP_LIBRARY_MAJOR_VERSION,
  HIP_LIBRARY_MINOR_VERSION,
  HIP_LIBRARY_PATCH_LEVEL
} hipLibraryPropertyType;

#elif !defined(__HIP_PLATFORM_AMD__) && defined(__HIP_PLATFORM_NVIDIA__)
#include "library_types.h"
#else
#error("Must define exactly one of __HIP_PLATFORM_AMD__ or __HIP_PLATFORM_NVIDIA__");
#endif

#endif
