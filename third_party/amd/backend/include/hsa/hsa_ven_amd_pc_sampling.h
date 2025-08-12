////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_VEN_AMD_PC_SAMPLING_H
#define HSA_VEN_AMD_PC_SAMPLING_H

#include "hsa.h"

#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/


/**
 * @brief HSA AMD Vendor PC Sampling APIs
 * EXPERIMENTAL: All PC Sampling APIs are currently in an experimental phase and the APIs may be
 * modified extensively in the future
 */

/**
 * @brief PC Sampling sample data for hosttrap sampling method
 */
typedef struct {
  uint64_t pc;
  uint64_t exec_mask;
  uint32_t workgroup_id_x;
  uint32_t workgroup_id_y;
  uint32_t workgroup_id_z;
  uint32_t wave_in_wg : 6;
  uint32_t chiplet    : 3;   // Currently not used
  uint32_t reserved   : 23;
  uint32_t hw_id;
  uint32_t reserved0;
  uint64_t reserved1;
  uint64_t timestamp;
  uint64_t correlation_id;
} perf_sample_hosttrap_v1_t;

/**
 * @brief PC Sampling sample data for stochastic sampling method
 */
typedef struct {
  uint64_t pc;
  uint64_t exec_mask;
  uint32_t workgroup_id_x;
  uint32_t workgroup_id_y;
  uint32_t workgroup_id_z;
  uint32_t wave_in_wg : 6;
  uint32_t chiplet    : 3;   // Currently not used
  uint32_t reserved   : 23;
  uint32_t hw_id;
  uint32_t perf_snapshot_data;
  uint32_t perf_snapshot_data1;
  uint32_t perf_snapshot_data2;
  uint64_t timestamp;
  uint64_t correlation_id;
} perf_sample_snapshot_v1_t;

/**
 * @brief PC Sampling method kinds
 */
typedef enum {
  HSA_VEN_AMD_PCS_METHOD_HOSTTRAP_V1,
  HSA_VEN_AMD_PCS_METHOD_STOCHASTIC_V1
} hsa_ven_amd_pcs_method_kind_t;

/**
 * @brief PC Sampling interval unit type
 */
typedef enum {
  HSA_VEN_AMD_PCS_INTERVAL_UNITS_MICRO_SECONDS,
  HSA_VEN_AMD_PCS_INTERVAL_UNITS_CLOCK_CYCLES,
  HSA_VEN_AMD_PCS_INTERVAL_UNITS_INSTRUCTIONS
} hsa_ven_amd_pcs_units_t;

/**
 * @brief HSA callback function to perform the copy onto a destination buffer
 *
 * If data_size is 0, HSA will stop current copy operation and keep remaining data in internal
 * buffers. Remaining contents of HSA internal buffers will be included in next
 * hsa_ven_amd_pcs_data_ready_callback_t. HSA internal buffers can also be drained by calling
 * hsa_ven_amd_pcs_flush.
 *
 * @param[in] hsa_callback_data private data to pass back to HSA. Provided in
 * hsa_ven_amd_pcs_data_ready_callback_t
 *
 * @param[in] data_size size of destination buffer in bytes.
 * @param[in] destination destination buffer
 * @retval    TBD: but could be used to indicate that there is no more data to be read.
 * Or indicate an error and abort of current copy operations
 */
typedef hsa_status_t (*hsa_ven_amd_pcs_data_copy_callback_t)(void* hsa_callback_data,
                                                             size_t data_size, void* destination);

/**
 * @brief HSA callback function to to indicate that there is data ready to be copied
 *
 * When the client receives this callback, the client should call back @p data_copy_callback for HSA
 * to perform the copy operation into an available buffer. @p data_copy_callback can be called back
 * multiple times with smaller @p data_size to split the copy operation.
 *
 * This callback must not call ::hsa_ven_amd_pcs_flush.
 *
 * @param[in] client_callback_data client private data passed in via
 * hsa_ven_amd_pcs_create/hsa_ven_amd_pcs_create_from_id
 * @param[in] data_size size of data available to be copied
 * @param[in] lost_sample_count number of lost samples since last call to
 * hsa_ven_amd_pcs_data_ready_callback_t.
 * @param[in] data_copy_callback callback function for HSA to perform the actual copy
 * @param[in] hsa_callback_data private data to pass back to HSA
 */
typedef void (*hsa_ven_amd_pcs_data_ready_callback_t)(
    void* client_callback_data, size_t data_size, size_t lost_sample_count,
    hsa_ven_amd_pcs_data_copy_callback_t data_copy_callback, void* hsa_callback_data);

/**
 * @brief Opaque handle representing a sampling session.
 * Two sessions having same handle value represent the same session
 */
typedef struct {
  uint64_t handle;
} hsa_ven_amd_pcs_t;

/**
 * @brief PC Sampling configuration flag options
 */
typedef enum {
  /* The interval for this sampling method have to be a power of 2 */
  HSA_VEN_AMD_PCS_CONFIGURATION_FLAGS_INTERVAL_POWER_OF_2 = (1 << 0)
} hsa_ven_amd_pcs_configuration_flags_t;

/**
 * @brief PC Sampling method information
 * Used to provide client with list of supported PC Sampling methods
 */
typedef struct {
  hsa_ven_amd_pcs_method_kind_t method;
  hsa_ven_amd_pcs_units_t units;
  size_t min_interval;
  size_t max_interval;
  uint64_t flags;
} hsa_ven_amd_pcs_configuration_t;

/**
 * @brief Callback function to iterate through list of supported PC Sampling configurations
 *
 * @param[in] configuration one entry for supported PC Sampling method and configuration options
 * @param[in] callback_data client private callback data that was passed in when calling
 * hsa_ven_amd_pcs_iterate_configuration
 */
typedef hsa_status_t (*hsa_ven_amd_pcs_iterate_configuration_callback_t)(
    const hsa_ven_amd_pcs_configuration_t* configuration, void* callback_data);

/**
 * @brief Iterate through list of current supported PC Sampling configurations for this @p agent
 *
 * HSA will callback @p configuration_callback for each currently available PC Sampling
 * configuration. The list of currently available configurations may not be the complete list of
 * configurations supported on the @p agent. The list of currently available configurations may be
 * reduced if the @p agent is currently handling other PC sampling sessions.
 *
 * @param[in] agent target agent
 * @param[in] configuration_callback callback function to iterate through list of configurations
 * @param[in] callback_data client private callback data
 **/
hsa_status_t hsa_ven_amd_pcs_iterate_configuration(
    hsa_agent_t agent, hsa_ven_amd_pcs_iterate_configuration_callback_t configuration_callback,
    void* callback_data);

/**
 * @brief  Create a PC Sampling session on @p agent
 *
 * Allocate the resources required for a PC Sampling session. The @p method, @p units, @p interval
 * parameters must be a legal configuration value, as described by the
 * hsa_ven_amd_pcs_configuration_t configurations passed to the callbacks of
 * hsa_ven_amd_pcs_iterate_configuration for this @p agent.
 * A successfull call may restrict the list of possible PC sampling methods available to subsequent
 * calls to hsa_ven_amd_pcs_iterate_configuration on the same agent as agents have limitations
 * on what types of PC sampling they can perform concurrently.
 * For all successful calls, hsa_ven_amd_pcs_destroy should be called to free this session.
 * The session will be in a stopped/inactive state after this call
 *
 * @param[in] agent target agent
 * @param[in] method method to use
 * @param[in] units sampling units
 * @param[in] interval sampling interval in @p units
 * @param[in] latency expected latency in microseconds for client to provide a buffer for the data
 * copy callback once HSA calls @p data_ready_callback. This is a performance hint to avoid the
 * buffer filling up before the client is notified that data is ready. HSA-runtime will estimate
 * how many samples are received within @p latency and call @p data_ready_callback ahead of time so
 * that the client has @p latency time to allocate the buffer before the HSA-runtime internal
 * buffers are full. The value of latency can be 0.
 * @param[in] buffer_size size of client buffer in bytes. @p data_ready_callback will be called once
 * HSA-runtime has enough samples to fill @p buffer_size. This needs to be a multiple of size of
 * perf_sample_hosttrap_v1_t or size of perf_sample_snapshot_v1_t.
 * @param[in] data_ready_callback client callback function that will be called when:
 *   1. There is enough samples fill a buffer with @p buffer_size  - estimated samples received
 *      within @p latency period.
 * OR
 *   2. When hsa_ven_amd_pcs_flush is called.
 * @param[in] client_callback_data client private data to be provided back when data_ready_callback
 * is called.
 * @param[out] pc_sampling PC sampling session handle used to reference this session when calling
 * hsa_ven_amd_pcs_start, hsa_ven_amd_pcs_stop, hsa_ven_amd_pcs_destroy
 *
 * @retval ::HSA_STATUS_SUCCESS session created successfully
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT invalid parameters
 * @retval ::HSA_STATUS_ERROR_RESOURCE_BUSY agent currently handling another PC Sampling session and
 * cannot handle the type requested.
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Failed to allocate resources
 * @retval ::HSA_STATUS_ERROR Unexpected error
 **/
hsa_status_t hsa_ven_amd_pcs_create(hsa_agent_t agent, hsa_ven_amd_pcs_method_kind_t method,
                                    hsa_ven_amd_pcs_units_t units, size_t interval, size_t latency,
                                    size_t buffer_size,
                                    hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback,
                                    void* client_callback_data, hsa_ven_amd_pcs_t* pc_sampling);


/**
 * @brief  Creates a PC Sampling session on @p agent. Assumes that the caller provides the
 * @p pcs_id generated by the previous call to the underlying driver that reserved PC sampling
 * on the @p agent.
 *
 * Similar to the @ref hsa_ven_amd_pcs_create with the difference that it inherits an existing
 * PC sampling session that was previously created in the underlying driver.
 *
 * Allocate the resources required for a PC Sampling session. The @p method, @p units, @p interval
 * parameters must be a legal configuration value, and match the parameters that we used to create
 * the underlying PC Sampling session in the underlying driver.
 * A successfull call may restrict the list of possible PC sampling methods available to subsequent
 * calls to hsa_ven_amd_pcs_iterate_configuration on the same agent as agents have limitations
 * on what types of PC sampling they can perform concurrently.
 * For all successful calls, hsa_ven_amd_pcs_destroy should be called to free this session.
 * The session will be in a stopped/inactive state after this call
 *
 * @param[in] pcs_id ID that uniquely identifies the PC sampling session within underlying driver
 * @param[in] agent target agent
 * @param[in] method method to use
 * @param[in] units sampling units
 * @param[in] interval sampling interval in @p units
 * @param[in] latency expected latency in microseconds for client to provide a buffer for the data
 * copy callback once HSA calls @p data_ready_callback. This is a performance hint to avoid the
 * buffer filling up before the client is notified that data is ready. HSA-runtime will estimate
 * how many samples are received within @p latency and call @p data_ready_callback ahead of time so
 * that the client has @p latency time to allocate the buffer before the HSA-runtime internal
 * buffers are full. The value of latency can be 0.
 * @param[in] buffer_size size of client buffer in bytes. @p data_ready_callback will be called once
 * HSA-runtime has enough samples to fill @p buffer_size. This needs to be a multiple of size of
 * perf_sample_hosttrap_v1_t or size of perf_sample_snapshot_v1_t.
 * @param[in] data_ready_callback client callback function that will be called when:
 *   1. There is enough samples fill a buffer with @p buffer_size  - estimated samples received
 *      within @p latency period.
 * OR
 *   2. When hsa_ven_amd_pcs_flush is called.
 * @param[in] client_callback_data client private data to be provided back when data_ready_callback
 * is called.
 * @param[out] pc_sampling PC sampling session handle used to reference this session when calling
 * hsa_ven_amd_pcs_start, hsa_ven_amd_pcs_stop, hsa_ven_amd_pcs_destroy
 *
 * @retval ::HSA_STATUS_SUCCESS session created successfully
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT invalid parameters
 * @retval ::HSA_STATUS_ERROR_RESOURCE_BUSY agent currently handling another PC Sampling session and
 * cannot handle the type requested.
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Failed to allocate resources
 * @retval ::HSA_STATUS_ERROR Unexpected error
 **/
hsa_status_t hsa_ven_amd_pcs_create_from_id(
    uint32_t pcs_id, hsa_agent_t agent, hsa_ven_amd_pcs_method_kind_t method,
    hsa_ven_amd_pcs_units_t units, size_t interval, size_t latency, size_t buffer_size,
    hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback, void* client_callback_data,
    hsa_ven_amd_pcs_t* pc_sampling);

/**
 * @brief  Free a PC Sampling session on @p agent
 *
 * Free all the resources allocated for a PC Sampling session on @p agent
 * Internal buffers for this session will be lost.
 * If the session was active, the session will be stopped before it is destroyed.
 *
 * @param[in] pc_sampling PC sampling session handle
 *
 * @retval ::HSA_STATUS_SUCCESS Session destroyed successfully
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid PC sampling handle
 * @retval ::HSA_STATUS_ERROR unexpected error
 */
hsa_status_t hsa_ven_amd_pcs_destroy(hsa_ven_amd_pcs_t pc_sampling);

/**
 * @brief  Start a PC Sampling session
 *
 * Activate a PC Sampling session that was previous created.
 * The session with be in a active state after this call
 * If the session was already active, this will result in a no-op and will return HSA_STATUS_SUCCESS
 *
 * @param[in] pc_sampling PC sampling session handle
 *
 * @retval ::HSA_STATUS_SUCCESS Session started successfully
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid PC sampling handle
 * @retval ::HSA_STATUS_ERROR unexpected error
 */
hsa_status_t hsa_ven_amd_pcs_start(hsa_ven_amd_pcs_t pc_sampling);

/**
 * @brief  Stop a PC Sampling session
 *
 * Stop a session that is currently active
 * After a session is stopped HSA may still have some PC Sampling data in its internal buffers.
 * The internal buffers can be drained using hsa_ven_amd_pcs_flush. If the internal
 * buffers are not drained and the session is started again, the internal buffers will be available
 * on the next data_ready_callback.
 * If the session was already inactive, this will result in a no-op and will return
 * HSA_STATUS_SUCCESS
 *
 * @param[in] pc_sampling PC sampling session handle
 *
 * @retval ::HSA_STATUS_SUCCESS Session stopped successfully
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid PC sampling handle
 */
hsa_status_t hsa_ven_amd_pcs_stop(hsa_ven_amd_pcs_t pc_sampling);

/**
 * @brief  Flush internal buffers for a PC Sampling session
 *
 * Drain internal buffers for a PC Sampling session. If internal buffers have available data,
 * this trigger a data_ready_callback.
 *
 * The function blocks until all PC samples associated with the @p pc_sampling session
 * generated prior to the function call have been communicated by invocations of
 * @p data_ready_callback having completed execution.
 *
 * @param[in] pc_sampling PC sampling session handle
 *
 * @retval ::HSA_STATUS_SUCCESS Session flushed successfully
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid PC sampling handle
 */
hsa_status_t hsa_ven_amd_pcs_flush(hsa_ven_amd_pcs_t pc_sampling);

#define hsa_ven_amd_pc_sampling_1_00

/**
 * @brief The function pointer table for the PC Sampling v1.00 extension. Can be returned by
 * ::hsa_system_get_extension_table or ::hsa_system_get_major_extension_table.
 */
typedef struct hsa_ven_amd_pc_sampling_1_00_pfn_t {
  hsa_status_t (*hsa_ven_amd_pcs_iterate_configuration)(
      hsa_agent_t agent, hsa_ven_amd_pcs_iterate_configuration_callback_t configuration_callback,
      void* callback_data);

  hsa_status_t (*hsa_ven_amd_pcs_create)(hsa_agent_t agent, hsa_ven_amd_pcs_method_kind_t method,
                                         hsa_ven_amd_pcs_units_t units, size_t interval,
                                         size_t latency, size_t buffer_size,
                                         hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback,
                                         void* client_callback_data,
                                         hsa_ven_amd_pcs_t* pc_sampling);

  hsa_status_t (*hsa_ven_amd_pcs_create_from_id)(
      uint32_t pcs_id, hsa_agent_t agent, hsa_ven_amd_pcs_method_kind_t method,
      hsa_ven_amd_pcs_units_t units, size_t interval, size_t latency, size_t buffer_size,
      hsa_ven_amd_pcs_data_ready_callback_t data_ready_callback, void* client_callback_data,
      hsa_ven_amd_pcs_t* pc_sampling);

  hsa_status_t (*hsa_ven_amd_pcs_destroy)(hsa_ven_amd_pcs_t pc_sampling);

  hsa_status_t (*hsa_ven_amd_pcs_start)(hsa_ven_amd_pcs_t pc_sampling);

  hsa_status_t (*hsa_ven_amd_pcs_stop)(hsa_ven_amd_pcs_t pc_sampling);

  hsa_status_t (*hsa_ven_amd_pcs_flush)(hsa_ven_amd_pcs_t pc_sampling);

} hsa_ven_amd_pc_sampling_1_00_pfn_t;

#ifdef __cplusplus
}  // end extern "C" block
#endif /*__cplusplus*/

#endif /* HSA_VEN_AMD_PC_SAMPLING_H */
