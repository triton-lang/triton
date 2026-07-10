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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup EXTERNAL_CORRELATION External Correlation IDs
 * @brief User-defined correlation identifiers to supplement rocprofiler generated correlation ids
 *
 * @{
 */

/**
 * @brief (experimental) ROCProfiler External Correlation ID Operations.
 *
 * These kinds correspond to callback and buffered tracing kinds (@see
 * ::rocprofiler_callback_tracing_kind_t and ::rocprofiler_buffer_tracing_kind_t) which generate
 * correlation IDs. Typically, rocprofiler-sdk uses the most recent external correlation ID on the
 * current thread set via ::rocprofiler_push_external_correlation_id; however, this approach can be
 * problematic if a new external correlation ID should be set before the
 * ::ROCPROFILER_CALLBACK_PHASE_ENTER callback or if relevant external correlation IDs are desired
 * when the buffered tracing is used. Thus, rocprofiler-sdk provides a way for tools to get a
 * callback whenever an external correlation ID is needed. However, this can add significant
 * overhead for those who only need these callbacks for, say, kernel dispatches while the HSA API is
 * being traced (i.e. lots of callbacks for HSA API functions). The enumeration below is provided to
 * ensure that tools can default to using the external correlation IDs set via the push/pop methods
 * where the external correlation ID value is not important while also getting a request for an
 * external correlation ID for other tracing kinds.
 */
// NOLINTNEXTLINE(performance-enum-size)
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_external_correlation_id_request_kind_t
{
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_NONE = 0,               ///< Unknown kind
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_CORE_API,           ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_AMD_EXT_API,        ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_IMAGE_EXT_API,      ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HSA_FINALIZE_EXT_API,   ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API,        ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_COMPILER_API,       ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CORE_API,        ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CONTROL_API,     ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_NAME_API,        ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY,            ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH,        ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_SCRATCH_MEMORY,         ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_RCCL_API,               ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_OMPT,                   ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION,      ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCDECODE_API,          ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_ROCJPEG_API,            ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MARKER_CORE_RANGE_API,  ///<
    ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_LAST,
} rocprofiler_external_correlation_id_request_kind_t;

/**
 * @brief (experimental) Callback requesting a value for the external correlation id.
 *
 * @param [in] thread_id Id of the thread making the request
 * @param [in] context_id Id of the context making the request
 * @param [in] kind Origin of the external correlation id request
 * @param [in] operation Regardless of whether callback or buffer tracing is being used, the
 * operation value will be the same, i.e., regardless of whether callback kind is
 * ::ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API or the buffer record kind is
 * ::ROCPROFILER_BUFFER_TRACING_HSA_CORE_API, the data/record for `hsa_init` will have an operation
 * value of ::ROCPROFILER_HSA_CORE_API_ID_hsa_init.
 * @param [in] internal_corr_id_value Current internal correlation ID value for the request
 * @param [out] external_corr_id_value Set this value to the desired external correlation ID value
 * @param [in] data The `callback_args` value passed to
 * ::rocprofiler_configure_external_correlation_id_request_service.
 * @returns int
 * @retval 0 Used to indicate the tool had zero issues setting the external correlation ID field
 * @retval 1 (or any other non-zero number) Used to indicate the callback did not set an external
 * correlation ID value and the thread-local value for the most recently pushed external correlation
 * ID should be used instead
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef int (*rocprofiler_external_correlation_id_request_cb_t)(
    rocprofiler_thread_id_t                            thread_id,
    rocprofiler_context_id_t                           context_id,
    rocprofiler_external_correlation_id_request_kind_t kind,
    rocprofiler_tracing_operation_t                    operation,
    uint64_t                                           internal_corr_id_value,
    rocprofiler_user_data_t*                           external_corr_id_value,
    void*                                              data);

/**
 * @brief (experimental) Configure External Correlation ID Request Service.
 *
 * @param [in] context_id Context to associate the service with
 * @param [in] kinds Array of ::rocprofiler_external_correlation_id_request_kind_t values. If
 * this parameter is null, all tracing operations will invoke the callback to request an external
 * correlation ID.
 * @param [in] kinds_count If the kinds array is non-null, set this to the size of the
 * array.
 * @param [in] callback The function to invoke for an external correlation ID request
 * @param [in] callback_args Data provided to every invocation of the callback function
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED Invoked outside of the initialization
 * function in ::rocprofiler_tool_configure_result_t provided to rocprofiler via
 * ::rocprofiler_configure function
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND The provided context is not valid/registered
 * @retval ::ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED if the same
 * ::rocprofiler_callback_tracing_kind_t value is provided more than once (per context) -- in other
 * words, we do not support overriding or combining the kinds in separate function calls.
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_configure_external_correlation_id_request_service(
    rocprofiler_context_id_t                                  context_id,
    const rocprofiler_external_correlation_id_request_kind_t* kinds,
    size_t                                                    kinds_count,
    rocprofiler_external_correlation_id_request_cb_t          callback,
    void* callback_args) ROCPROFILER_API ROCPROFILER_NONNULL(4);

/**
 * @brief Query the name of the external correlation request kind. The name retrieved from this
 * function is a string literal that is encoded in the read-only section of the binary (i.e. it is
 * always "allocated" and never "deallocated").
 *
 * @param [in] kind External correlation id request domain
 * @param [out] name If non-null and the name is a constant string that does not require dynamic
 * allocation, this paramter will be set to the address of the string literal, otherwise it will
 * be set to nullptr
 * @param [out] name_len If non-null, this will be assigned the length of the name (regardless of
 * the name is a constant string or requires dynamic allocation)
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_KIND_NOT_FOUND Returned if the domain id is not valid
 * @retval ::ROCPROFILER_STATUS_SUCCESS Returned if a valid domain, regardless if there is a
 * constant string or not.
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_query_external_correlation_id_request_kind_name(
    rocprofiler_external_correlation_id_request_kind_t kind,
    const char**                                       name,
    uint64_t*                                          name_len) ROCPROFILER_API;

/**
 * @brief Push default value for `external` field in ::rocprofiler_correlation_id_t onto stack.
 *
 * External correlation ids are thread-local values. However, if rocprofiler internally requests an
 * external correlation id on a non-main thread and an external correlation id has not been pushed
 * for this thread, the external correlation ID will default to the latest external correlation id
 * on the main thread -- this allows tools to push an external correlation id once on the main
 * thread for, say, the MPI rank or process-wide UUID and this value will be used by all subsequent
 * child threads.
 *
 * @param [in] context Associated context
 * @param [in] tid thread identifier. @see rocprofiler_get_thread_id
 * @param [in] external_correlation_id User data to place in external field in
 * ::rocprofiler_correlation_id_t
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND Context does not exist
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT Thread id is not valid
 */
rocprofiler_status_t
rocprofiler_push_external_correlation_id(rocprofiler_context_id_t context,
                                         rocprofiler_thread_id_t  tid,
                                         rocprofiler_user_data_t  external_correlation_id)
    ROCPROFILER_API;

/**
 * @brief Pop default value for `external` field in ::rocprofiler_correlation_id_t off of stack.
 *
 * @param [in] context Associated context
 * @param [in] tid thread identifier. @see rocprofiler_get_thread_id
 * @param [out] external_correlation_id Correlation id data popped off the stack
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND Context does not exist
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT Thread id is not valid
 */
rocprofiler_status_t
rocprofiler_pop_external_correlation_id(rocprofiler_context_id_t context,
                                        rocprofiler_thread_id_t  tid,
                                        rocprofiler_user_data_t* external_correlation_id)
    ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
