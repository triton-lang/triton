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

/**
 * @brief ROCProfiler enumeration of Marker (ROCTx) API tracing operations
 */
typedef enum rocprofiler_marker_core_api_id_t // NOLINT(performance-enum-size)
{ ROCPROFILER_MARKER_CORE_API_ID_NONE = -1,
  ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA = 0,
  ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA,
  ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop,
  ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA,
  ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop,
  ROCPROFILER_MARKER_CORE_API_ID_roctxGetThreadId,
  ROCPROFILER_MARKER_CORE_API_ID_LAST,
} rocprofiler_marker_core_api_id_t;

typedef enum rocprofiler_marker_control_api_id_t // NOLINT(performance-enum-size)
{ ROCPROFILER_MARKER_CONTROL_API_ID_NONE = -1,
  ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause = 0,
  ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume,
  ROCPROFILER_MARKER_CONTROL_API_ID_LAST,
} rocprofiler_marker_control_api_id_t;

typedef enum rocprofiler_marker_name_api_id_t // NOLINT(performance-enum-size)
{ ROCPROFILER_MARKER_NAME_API_ID_NONE = -1,
  ROCPROFILER_MARKER_NAME_API_ID_roctxNameOsThread = 0,
  ROCPROFILER_MARKER_NAME_API_ID_roctxNameHsaAgent,
  ROCPROFILER_MARKER_NAME_API_ID_roctxNameHipDevice,
  ROCPROFILER_MARKER_NAME_API_ID_roctxNameHipStream,
  ROCPROFILER_MARKER_NAME_API_ID_LAST,
} rocprofiler_marker_name_api_id_t;

typedef enum rocprofiler_marker_core_range_api_id_t // NOLINT(performance-enum-size)
{ ROCPROFILER_MARKER_CORE_RANGE_API_ID_NONE = -1,
  ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxMarkA = 0,
  ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxThreadRangeA,
  ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxProcessRangeA,
  ROCPROFILER_MARKER_CORE_RANGE_API_ID_roctxGetThreadId,
  ROCPROFILER_MARKER_CORE_RANGE_API_ID_LAST,
} rocprofiler_marker_core_range_api_id_t;
