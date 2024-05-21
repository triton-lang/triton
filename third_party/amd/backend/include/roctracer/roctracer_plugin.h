/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

/** \section roctracer_plugin_api ROCtracer Plugin API
 *
 * The ROCtracer Plugin API is used by the ROCtracer Tool to output all tracing
 * information. Different implementations of the ROCtracer Plugin API can be
 * developed that output the tracing data in different formats.
 * The ROCtracer Tool can be configured to load a specific library that
 * supports the user desired format.
 *
 * The API is not thread safe. It is the responsibility of the ROCtracer Tool
 * to ensure the operations are synchronized and not called concurrently. There
 * is no requirement for the ROCtracer Tool to report trace data in any
 * specific order. If the format supported by plugin requires specific
 * ordering, it is the responsibility of the plugin implementation to perform
 * any necessary sorting.
 */

/**
 * \file
 * ROCtracer Tool Plugin API interface.
 */

#ifndef ROCTRACER_PLUGIN_H_
#define ROCTRACER_PLUGIN_H_

#include "roctracer.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/** \defgroup initialization_group Initialization and Finalization
 *
 * The ROCtracer Plugin API must be initialized before using any of the
 * operations to report trace data, and finalized after the last trace data has
 * been reported.
 *
 * @{
 */

/**
 * Initialize plugin.
 *
 * Must be called before any other operation.
 *
 * @param[in] roctracer_major_version The major version of the ROCtracer API
 * being used by the ROCtracer Tool. An error is reported if this does not
 * match the major version of the ROCtracer API used to build the plugin
 * library. This ensures compatibility of the trace data format.
 *
 * @param[in] roctracer_minor_version The minor version of the ROCtracer API
 * being used by the ROCtracer Tool. An error is reported if the
 * \p roctracer_major_version matches and this is greater than the minor
 * version of the ROCtracer API used to build the plugin library. This ensures
 * compatibility of the trace data format.
 *
 * @return Returns 0 on success and -1 on error.
 */
ROCTRACER_EXPORT int roctracer_plugin_initialize(
    uint32_t roctracer_major_version, uint32_t roctracer_minor_version);

/**
 * Finalize plugin.
 *
 * This must be called after ::roctracer_plugin_initialize and after all trace
 * data has been reported by ::roctracer_plugin_write_callback_record and
 * ::roctracer_plugin_write_activity_records.
 */
ROCTRACER_EXPORT void roctracer_plugin_finalize();

/** @} */

/** \defgroup trace_record_write_functions Trace data reporting
 *
 * Operations to output trace data.
 *
 * @{
 */

/**
 * Report a single callback trace data.
 *
 * @param[in] record Primarily domain independent trace data.
 *
 * @param[in] callback_data Domain specific trace data. The type of this
 * argument depends on the values of \p record.domain.
 *
 * @return Returns 0 on success and -1 on error.
 */
ROCTRACER_EXPORT int roctracer_plugin_write_callback_record(
    const roctracer_record_t* record, const void* callback_data);

/**
 * Report a range of activity trace data.
 *
 * Reports a range of primarily domain independent trace data. The range is
 * specified by a pointer to the first record and a pointer to one past the
 * last record. ::roctracer_next_record is used to iterate the range in forward
 * order.
 *
 * @param[in] begin Pointer to the first record.
 *
 * @param[in] end Pointer to one past the last record.
 *
 * @return Returns 0 on success and -1 on error.
 */
ROCTRACER_EXPORT int roctracer_plugin_write_activity_records(
    const roctracer_record_t* begin, const roctracer_record_t* end);

/** @} */

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif /* ROCTRACER_PLUGIN_H_ */
