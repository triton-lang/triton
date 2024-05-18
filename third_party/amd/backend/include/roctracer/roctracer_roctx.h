/* Copyright (c) 2018-2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#ifndef ROCTRACER_ROCTX_H_
#define ROCTRACER_ROCTX_H_

#include "roctx.h"

/**
 *  ROCTX API ID enumeration
 */
enum roctx_api_id_t {
  ROCTX_API_ID_roctxMarkA = 0,
  ROCTX_API_ID_roctxRangePushA = 1,
  ROCTX_API_ID_roctxRangePop = 2,
  ROCTX_API_ID_roctxRangeStartA = 3,
  ROCTX_API_ID_roctxRangeStop = 4,
  ROCTX_API_ID_NUMBER,
};

/**
 *  ROCTX callbacks data type
 */
typedef struct roctx_api_data_s {
  union {
    struct {
      const char* message;
      roctx_range_id_t id;
    };
    struct {
      const char* message;
    } roctxMarkA;
    struct {
      const char* message;
    } roctxRangePushA;
    struct {
      const char* message;
    } roctxRangePop;
    struct {
      const char* message;
      roctx_range_id_t id;
    } roctxRangeStartA;
    struct {
      const char* message;
      roctx_range_id_t id;
    } roctxRangeStop;
  } args;
} roctx_api_data_t;

#endif /* ROCTRACER_ROCTX_H_ */
