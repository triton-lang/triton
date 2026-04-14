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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>

#include <rocprofiler-sdk-roctx/api_trace.h>
#include <rocprofiler-sdk-roctx/types.h>

#include <stdint.h>

ROCPROFILER_EXTERN_C_INIT

// Empty struct has a size of 0 in C but size of 1 in C++.
// This struct is added to the union members which represent
// functions with no arguments to ensure ABI compatibility
typedef struct rocprofiler_marker_api_no_args {
  char empty;
} rocprofiler_marker_api_no_args;

typedef union rocprofiler_marker_api_retval_t {
  int32_t int32_t_retval;
  int64_t int64_t_retval;
  roctx_range_id_t roctx_range_id_t_retval;
} rocprofiler_marker_api_retval_t;

typedef union rocprofiler_marker_api_args_t {
  struct {
    const char *message;
  } roctxMarkA;
  struct {
    const char *message;
  } roctxRangePushA;
  struct {
    // Empty struct has a size of 0 in C but size of 1 in C++.
    // Add the rocprofiler_marker_api_no_args struct to fix this
    rocprofiler_marker_api_no_args no_args;
  } roctxRangePop;
  struct {
    const char *message;
  } roctxRangeStartA;
  struct {
    roctx_range_id_t id;
  } roctxRangeStop;
  struct {
    roctx_thread_id_t *tid;
  } roctxGetThreadId;
  struct {
    roctx_thread_id_t tid;
  } roctxProfilerPause;
  struct {
    roctx_thread_id_t tid;
  } roctxProfilerResume;
  struct {
    const char *name;
  } roctxNameOsThread;
  struct {
    const char *name;
    const struct hsa_agent_s *agent;
  } roctxNameHsaAgent;
  struct {
    const char *name;
    int device_id;
  } roctxNameHipDevice;
  struct {
    const char *name;
    const struct ihipStream_t *stream;
  } roctxNameHipStream;
  struct {
    const char *message;
  } roctxThreadRangeA;
  struct {
    const char *message;
    // roctx_range_id_t id;  // only set when range ends in callback tracing
  } roctxProcessRangeA;
} rocprofiler_marker_api_args_t;

ROCPROFILER_EXTERN_C_FINI
