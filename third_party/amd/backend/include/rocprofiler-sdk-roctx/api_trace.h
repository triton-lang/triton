// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "rocprofiler-sdk-roctx/defines.h"
#include "rocprofiler-sdk-roctx/types.h"
#include "rocprofiler-sdk-roctx/version.h"

#include <sched.h>
#include <stddef.h>
#include <stdint.h>

ROCTX_EXTERN_C_INIT

#define ROCTX_API_TABLE_VERSION_MAJOR 0
#define ROCTX_API_TABLE_VERSION_STEP 0

#define ROCTX_CORE_API_TABLE_VERSION_MAJOR 0
#define ROCTX_CORE_API_TABLE_VERSION_STEP 0

#define ROCTX_CONTROL_API_TABLE_VERSION_MAJOR 0
#define ROCTX_CONTROL_API_TABLE_VERSION_STEP 0

#define ROCTX_RESOURCE_API_TABLE_VERSION_MAJOR 0
#define ROCTX_RESOURCE_API_TABLE_VERSION_STEP 0

typedef uint64_t roctx_range_id_t;
typedef void (*roctxMarkA_fn_t)(const char *message);
typedef int (*roctxRangePushA_fn_t)(const char *message);
typedef int (*roctxRangePop_fn_t)(void);
typedef roctx_range_id_t (*roctxRangeStartA_fn_t)(const char *message);
typedef void (*roctxRangeStop_fn_t)(roctx_range_id_t id);
typedef int (*roctxProfilerPause_fn_t)(roctx_thread_id_t tid);
typedef int (*roctxProfilerResume_fn_t)(roctx_thread_id_t tid);
typedef int (*roctxNameOsThread_fn_t)(const char *name);
typedef int (*roctxNameHsaAgent_fn_t)(const char *name,
                                      const struct hsa_agent_s *agent);
typedef int (*roctxNameHipDevice_fn_t)(const char *name, int device_id);
typedef int (*roctxNameHipStream_fn_t)(const char *name,
                                       const struct ihipStream_t *stream);
typedef int (*roctxGetThreadId_fn_t)(roctx_thread_id_t *tid);

typedef struct roctxCoreApiTable_t {
  uint64_t size;
  roctxMarkA_fn_t roctxMarkA_fn;
  roctxRangePushA_fn_t roctxRangePushA_fn;
  roctxRangePop_fn_t roctxRangePop_fn;
  roctxRangeStartA_fn_t roctxRangeStartA_fn;
  roctxRangeStop_fn_t roctxRangeStop_fn;
  roctxGetThreadId_fn_t roctxGetThreadId_fn;
} roctxCoreApiTable_t;

typedef struct roctxControlApiTable_t {
  uint64_t size;
  roctxProfilerPause_fn_t roctxProfilerPause_fn;
  roctxProfilerResume_fn_t roctxProfilerResume_fn;
} roctxControlApiTable_t;

typedef struct roctxNameApiTable_t {
  uint64_t size;
  roctxNameOsThread_fn_t roctxNameOsThread_fn;
  roctxNameHsaAgent_fn_t roctxNameHsaAgent_fn;
  roctxNameHipDevice_fn_t roctxNameHipDevice_fn;
  roctxNameHipStream_fn_t roctxNameHipStream_fn;
} roctxNameApiTable_t;

ROCTX_EXTERN_C_FINI
