////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
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

// HSA AMD extension.

#ifndef HSA_RUNTIME_EXT_AMD_H_
#define HSA_RUNTIME_EXT_AMD_H_

#include "hsa.h"
#include "hsa_ext_image.h"
#include "hsa_ven_amd_pc_sampling.h"

/**
 * - 1.0 - initial version
 * - 1.1 - dmabuf export
 * - 1.2 - hsa_amd_memory_async_copy_on_engine
 * - 1.3 - HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED pool
 * - 1.4 - Virtual Memory API
 * - 1.5 - hsa_amd_agent_info: HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES
 * - 1.6 - Virtual Memory API: hsa_amd_vmem_address_reserve_align
 */
#define HSA_AMD_INTERFACE_VERSION_MAJOR 1
#define HSA_AMD_INTERFACE_VERSION_MINOR 6

#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup aql Architected Queuing Language
 *  @{
 */

/**
 * @brief Macro to use to determine that a  flag is set when querying flags within uint8_t[8]
 * types
 */
static __inline__ __attribute__((always_inline)) bool hsa_flag_isset64(uint8_t* value,
                                                                       uint32_t bit) {
  unsigned int index = bit / 8;
  unsigned int subBit = bit % 8;
  return ((uint8_t*)value)[index] & (1 << subBit);
}

/**
 * @brief A fixed-size type used to represent ::hsa_signal_condition_t constants.
 */
typedef uint32_t hsa_signal_condition32_t;

/**
 * @brief AMD vendor specific packet type.
 */
typedef enum {
  /**
   * Packet used by agents to delay processing of subsequent packets until a
   * configurable condition is satisfied by an HSA signal.  Only kernel dispatch
   * queues created from AMD GPU Agents support this packet.
   */
  HSA_AMD_PACKET_TYPE_BARRIER_VALUE = 2,
} hsa_amd_packet_type_t;

/**
 * @brief A fixed-size type used to represent ::hsa_amd_packet_type_t constants.
 */
typedef uint8_t hsa_amd_packet_type8_t;

/**
 * @brief AMD vendor specific AQL packet header
 */
typedef struct hsa_amd_packet_header_s {
  /**
   * Packet header. Used to configure multiple packet parameters such as the
   * packet type. The parameters are described by ::hsa_packet_header_t.
   */
  uint16_t header;

  /**
   *Format of the vendor specific packet.
   */
  hsa_amd_packet_type8_t AmdFormat;

  /**
   * Reserved. Must be 0.
   */
  uint8_t reserved;
} hsa_amd_vendor_packet_header_t;

/**
 * @brief AMD barrier value packet.  Halts packet processing and waits for
 * (signal_value & ::mask) ::cond ::value to be satisfied, where signal_value
 * is the value of the signal ::signal.
 */
typedef struct hsa_amd_barrier_value_packet_s {
  /**
   * AMD vendor specific packet header.
   */
  hsa_amd_vendor_packet_header_t header;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved0;

  /**
   * Dependent signal object. A signal with a handle value of 0 is
   * allowed and is interpreted by the packet processor a satisfied
   * dependency.
   */
  hsa_signal_t signal;

  /**
   * Value to compare against.
   */
  hsa_signal_value_t value;

  /**
   * Bit mask to be combined by bitwise AND with ::signal's value.
   */
  hsa_signal_value_t mask;

  /**
   * Comparison operation.  See ::hsa_signal_condition_t.
   */
  hsa_signal_condition32_t cond;

  /**
   * Reserved. Must be 0.
   */
  uint32_t reserved1;

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved2;

  /**
   * Reserved. Must be 0.
   */
  uint64_t reserved3;

  /**
   * Signal used to indicate completion of the job. The application can use the
   * special signal handle 0 to indicate that no signal is used.
   */
  hsa_signal_t completion_signal;
} hsa_amd_barrier_value_packet_t;

/** @} */

/**
 * @brief Enumeration constants added to ::hsa_status_t.
 *
 * @remark Additions to hsa_status_t
 */
enum {
  /**
   * The memory pool is invalid.
   */
  HSA_STATUS_ERROR_INVALID_MEMORY_POOL = 40,

  /**
   * Agent accessed memory beyond the maximum legal address.
   */
  HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION = 41,

  /**
   * Agent executed an invalid shader instruction.
   */
  HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION = 42,

  /**
   * Agent attempted to access an inaccessible address.
   * See hsa_amd_register_system_event_handler and
   * HSA_AMD_GPU_MEMORY_FAULT_EVENT for more information on illegal accesses.
   */
  HSA_STATUS_ERROR_MEMORY_FAULT = 43,

  /**
   * The CU mask was successfully set but the mask attempted to enable a CU
   * which was disabled for the process.  CUs disabled for the process remain
   * disabled.
   */
  HSA_STATUS_CU_MASK_REDUCED = 44,

  /**
   * Exceeded number of VGPRs available on this agent
   */
  HSA_STATUS_ERROR_OUT_OF_REGISTERS = 45,

  /**
   * Resource is busy or temporarily unavailable
   */
  HSA_STATUS_ERROR_RESOURCE_BUSY = 46,
};

/**
 * @brief IOMMU version supported
 */
typedef enum {
  /**
   * IOMMU not supported
   */
  HSA_IOMMU_SUPPORT_NONE = 0,
  /* IOMMU V1 support is not relevant to user applications, so not reporting it */
  /**
   * IOMMU V2 supported
   */
  HSA_IOMMU_SUPPORT_V2 = 1,
} hsa_amd_iommu_version_t;

/**
 * @brief Agent attributes.
 */
typedef enum hsa_amd_agent_info_s {
  /**
   * Chip identifier. The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_CHIP_ID = 0xA000,
  /**
   * Size of a cacheline in bytes. The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_CACHELINE_SIZE = 0xA001,
  /**
   * The number of compute unit available in the agent. The type of this
   * attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT = 0xA002,
  /**
   * The maximum clock frequency of the agent in MHz. The type of this
   * attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_MAX_CLOCK_FREQUENCY = 0xA003,
  /**
   * Internal driver node identifier. The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_DRIVER_NODE_ID = 0xA004,
  /**
   * Max number of watch points on memory address ranges to generate exception
   * events when the watched addresses are accessed.  The type of this
   * attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_MAX_ADDRESS_WATCH_POINTS = 0xA005,
  /**
   * Agent BDF_ID, named LocationID in thunk. The type of this attribute is
   * uint32_t.
   */
  HSA_AMD_AGENT_INFO_BDFID = 0xA006,
  /**
   * Memory Interface width, the return value type is uint32_t.
   * This attribute is deprecated.
   */
  HSA_AMD_AGENT_INFO_MEMORY_WIDTH = 0xA007,
  /**
   * Max Memory Clock, the return value type is uint32_t.
   */
  HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY = 0xA008,
  /**
   * Board name of Agent - populated from MarketingName of Kfd Node
   * The value is an Ascii string of 64 chars.
   */
  HSA_AMD_AGENT_INFO_PRODUCT_NAME = 0xA009,
  /**
   * Maximum number of waves possible in a Compute Unit.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU = 0xA00A,
  /**
   * Number of SIMD's per compute unit CU
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_NUM_SIMDS_PER_CU = 0xA00B,
  /**
   * Number of Shader Engines (SE) in Gpu
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES = 0xA00C,
  /**
   * Number of Shader Arrays Per Shader Engines in Gpu
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE = 0xA00D,
  /**
   * Address of the HDP flush registers.  Use of these registers does not conform to the HSA memory
   * model and should be treated with caution.
   * The type of this attribute is hsa_amd_hdp_flush_t.
   */
  HSA_AMD_AGENT_INFO_HDP_FLUSH = 0xA00E,
  /**
   * PCIe domain for the agent.  Pairs with HSA_AMD_AGENT_INFO_BDFID
   * to give the full physical location of the Agent.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_DOMAIN = 0xA00F,
  /**
   * Queries for support of cooperative queues.  See ::HSA_QUEUE_TYPE_COOPERATIVE.
   * The type of this attribute is bool.
   */
  HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES = 0xA010,
  /**
   * Queries UUID of an agent. The value is an Ascii string with a maximum
   * of 21 chars including NUL. The string value consists of two parts: header
   * and body. The header identifies device type (GPU, CPU, DSP) while body
   * encodes UUID as a 16 digit hex string
   *
   * Agents that do not support UUID will return the string "GPU-XX" or
   * "CPU-XX" or "DSP-XX" depending upon their device type ::hsa_device_type_t
   */
  HSA_AMD_AGENT_INFO_UUID = 0xA011,
  /**
   * Queries for the ASIC revision of an agent. The value is an integer that
   * increments for each revision. This can be used by user-level software to
   * change how it operates, depending on the hardware version. This allows
   * selective workarounds for hardware errata.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_ASIC_REVISION = 0xA012,
  /**
   * Queries whether or not the host can directly access SVM memory that is
   * physically resident in the agent's local memory.
   * The type of this attribute is bool.
   */
  HSA_AMD_AGENT_INFO_SVM_DIRECT_HOST_ACCESS = 0xA013,
  /**
   * Some processors support more CUs than can reliably be used in a cooperative
   * dispatch.  This queries the count of CUs which are fully enabled for
   * cooperative dispatch.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_COOPERATIVE_COMPUTE_UNIT_COUNT = 0xA014,
  /**
   * Queries the amount of memory available in bytes accross all global pools
   * owned by the agent.
   * The type of this attribute is uint64_t.
   */
  HSA_AMD_AGENT_INFO_MEMORY_AVAIL = 0xA015,
  /**
   * Timestamp value increase rate, in Hz. The timestamp (clock) frequency is
   * in the range 1-400MHz.
   * The type of this attribute is uint64_t.
   */
  HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY = 0xA016,
  /**
   * Queries for the ASIC family ID of an agent.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_ASIC_FAMILY_ID = 0xA107,
  /**
   * Queries for the Packet Processor(CP Firmware) ucode version of an agent.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_UCODE_VERSION = 0xA108,
  /**
   * Queries for the SDMA engine ucode of an agent.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_SDMA_UCODE_VERSION = 0xA109,
  /**
   * Queries the number of SDMA engines.
   * If HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG query returns non-zero,
   * this query returns the number of SDMA engines optimized for
   * host to device bidirectional traffic.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_NUM_SDMA_ENG = 0xA10A,
  /**
   * Queries the number of additional SDMA engines optimized for D2D xGMI copies.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_NUM_SDMA_XGMI_ENG = 0xA10B,
  /**
   * Queries for version of IOMMU supported by agent.
   * The type of this attribute is hsa_amd_iommu_version_t.
   */
  HSA_AMD_AGENT_INFO_IOMMU_SUPPORT = 0xA110,
  /**
   * Queries for number of XCCs within the agent.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_NUM_XCC = 0xA111,
  /**
   * Queries for driver unique identifier.
   * The type of this attribute is uint32_t.
   */
  HSA_AMD_AGENT_INFO_DRIVER_UID = 0xA112,
  /**
   * Returns the hsa_agent_t of the nearest CPU agent
   * The type of this attribute is hsa_agent_t.
   */
  HSA_AMD_AGENT_INFO_NEAREST_CPU = 0xA113,
  /**
   * Bit-mask indicating memory properties of this agent. A memory property is set if the flag bit
   * is set at that position. User may use the hsa_flag_isset64 macro to verify whether a flag
   * is set. The type of this attribute is uint8_t[8].
   */
  HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES = 0xA114,
  /**
   * Bit-mask indicating AQL Extensions supported by this agent. An AQL extension is set if the flag
   * bit is set at that position. User may use the hsa_flag_isset64 macro to verify whether a flag
   * is set. The type of this attribute is uint8_t[8].
   */
  HSA_AMD_AGENT_INFO_AQL_EXTENSIONS = 0xA115 /* Not implemented yet */
} hsa_amd_agent_info_t;

/**
 * @brief Agent memory properties attributes
 */
typedef enum hsa_amd_agent_memory_properties_s {
  HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU = (1 << 0),
} hsa_amd_agent_memory_properties_t;

/**
 * @brief SDMA engine IDs unique by single set bit position.
 */
typedef enum hsa_amd_sdma_engine_id {
  HSA_AMD_SDMA_ENGINE_0 = 0x1,
  HSA_AMD_SDMA_ENGINE_1 = 0x2,
  HSA_AMD_SDMA_ENGINE_2 = 0x4,
  HSA_AMD_SDMA_ENGINE_3 = 0x8,
  HSA_AMD_SDMA_ENGINE_4 = 0x10,
  HSA_AMD_SDMA_ENGINE_5 = 0x20,
  HSA_AMD_SDMA_ENGINE_6 = 0x40,
  HSA_AMD_SDMA_ENGINE_7 = 0x80,
  HSA_AMD_SDMA_ENGINE_8 = 0x100,
  HSA_AMD_SDMA_ENGINE_9 = 0x200,
  HSA_AMD_SDMA_ENGINE_10 = 0x400,
  HSA_AMD_SDMA_ENGINE_11 = 0x800,
  HSA_AMD_SDMA_ENGINE_12 = 0x1000,
  HSA_AMD_SDMA_ENGINE_13 = 0x2000,
  HSA_AMD_SDMA_ENGINE_14 = 0x4000,
  HSA_AMD_SDMA_ENGINE_15 = 0x8000
} hsa_amd_sdma_engine_id_t;

typedef struct hsa_amd_hdp_flush_s {
  uint32_t* HDP_MEM_FLUSH_CNTL;
  uint32_t* HDP_REG_FLUSH_CNTL;
} hsa_amd_hdp_flush_t;

/**
 * @brief Region attributes.
 */
typedef enum hsa_amd_region_info_s {
  /**
   * Determine if host can access the region. The type of this attribute
   * is bool.
   */
  HSA_AMD_REGION_INFO_HOST_ACCESSIBLE = 0xA000,
  /**
   * Base address of the region in flat address space.
   */
  HSA_AMD_REGION_INFO_BASE = 0xA001,
  /**
   * Memory Interface width, the return value type is uint32_t.
   * This attribute is deprecated. Use HSA_AMD_AGENT_INFO_MEMORY_WIDTH.
   */
  HSA_AMD_REGION_INFO_BUS_WIDTH = 0xA002,
  /**
   * Max Memory Clock, the return value type is uint32_t.
   * This attribute is deprecated. Use HSA_AMD_AGENT_INFO_MEMORY_MAX_FREQUENCY.
   */
  HSA_AMD_REGION_INFO_MAX_CLOCK_FREQUENCY = 0xA003,
} hsa_amd_region_info_t;

/**
 * @brief Coherency attributes of fine grain region.
 */
typedef enum hsa_amd_coherency_type_s {
  /**
   * Coherent region.
   */
  HSA_AMD_COHERENCY_TYPE_COHERENT = 0,
  /**
   * Non coherent region.
   */
  HSA_AMD_COHERENCY_TYPE_NONCOHERENT = 1
} hsa_amd_coherency_type_t;

/**
 * @brief Get the coherency type of the fine grain region of an agent.
 *
 * @param[in] agent A valid agent.
 *
 * @param[out] type Pointer to a memory location where the HSA runtime will
 * store the coherency type of the fine grain region.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p type is NULL.
 */
hsa_status_t HSA_API hsa_amd_coherency_get_type(hsa_agent_t agent,
                                                hsa_amd_coherency_type_t* type);

/**
 * @brief Set the coherency type of the fine grain region of an agent.
 * Deprecated.  This is supported on KV platforms.  For backward compatibility
 * other platforms will spuriously succeed.
 *
 * @param[in] agent A valid agent.
 *
 * @param[in] type The coherency type to be set.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p type is invalid.
 */
hsa_status_t HSA_API hsa_amd_coherency_set_type(hsa_agent_t agent,
                                                hsa_amd_coherency_type_t type);

/**
 * @brief Structure containing profiling dispatch time information.
 *
 * Times are reported as ticks in the domain of the HSA system clock.
 * The HSA system clock tick and frequency is obtained via hsa_system_get_info.
 */
typedef struct hsa_amd_profiling_dispatch_time_s {
  /**
   * Dispatch packet processing start time.
   */
  uint64_t start;
  /**
   * Dispatch packet completion time.
   */
  uint64_t end;
} hsa_amd_profiling_dispatch_time_t;

/**
 * @brief Structure containing profiling async copy time information.
 *
 * Times are reported as ticks in the domain of the HSA system clock.
 * The HSA system clock tick and frequency is obtained via hsa_system_get_info.
 */
typedef struct hsa_amd_profiling_async_copy_time_s {
  /**
   * Async copy processing start time.
   */
  uint64_t start;
  /**
   * Async copy completion time.
   */
  uint64_t end;
} hsa_amd_profiling_async_copy_time_t;

/**
 * @brief Enable or disable profiling capability of a queue.
 *
 * @param[in] queue A valid queue.
 *
 * @param[in] enable 1 to enable profiling. 0 to disable profiling.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_QUEUE The queue is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p queue is NULL.
 */
hsa_status_t HSA_API
    hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int enable);

/**
 * @brief Enable or disable asynchronous memory copy profiling.
 *
 * @details The runtime will provide the copy processing start timestamp and
 * completion timestamp of each call to hsa_amd_memory_async_copy if the
 * async copy profiling is enabled prior to the call to
 * hsa_amd_memory_async_copy. The completion signal object is used to
 * hold the last async copy start and end timestamp. The client can retrieve
 * these timestamps via call to hsa_amd_profiling_get_async_copy_time.
 *
 * @param[in] enable True to enable profiling. False to disable profiling.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Failed on allocating resources
 * needed to profile the asynchronous copy.
 */
hsa_status_t HSA_API
    hsa_amd_profiling_async_copy_enable(bool enable);

/**
 * @brief Retrieve packet processing time stamps.
 *
 * @param[in] agent The agent with which the signal was last used.  For
 * instance, if the profiled dispatch packet is dispatched onto queue Q,
 * which was created on agent A, then this parameter must be A.
 *
 * @param[in] signal A signal used as the completion signal of the dispatch
 * packet to retrieve time stamps from.  This dispatch packet must have been
 * issued to a queue with profiling enabled and have already completed.  Also
 * the signal must not have yet been used in any other packet following the
 * completion of the profiled dispatch packet.
 *
 * @param[out] time Packet processing timestamps in the HSA system clock
 * domain.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SIGNAL The signal is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p time is NULL.
 */
hsa_status_t HSA_API hsa_amd_profiling_get_dispatch_time(
    hsa_agent_t agent, hsa_signal_t signal,
    hsa_amd_profiling_dispatch_time_t* time);

/**
 * @brief Retrieve asynchronous copy timestamps.
 *
 * @details Async copy profiling is enabled via call to
 * hsa_amd_profiling_async_copy_enable.
 *
 * @param[in] signal A signal used as the completion signal of the call to
 * hsa_amd_memory_async_copy.
 *
 * @param[out] time Async copy processing timestamps in the HSA system clock
 * domain.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SIGNAL The signal is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p time is NULL.
 */
hsa_status_t HSA_API hsa_amd_profiling_get_async_copy_time(
    hsa_signal_t signal, hsa_amd_profiling_async_copy_time_t* time);

/**
 * @brief Computes the frequency ratio and offset between the agent clock and
 * HSA system clock and converts the agent's tick to HSA system domain tick.
 *
 * @param[in] agent The agent used to retrieve the agent_tick. It is user's
 * responsibility to make sure the tick number is from this agent, otherwise,
 * the behavior is undefined.
 *
 * @param[in] agent_tick The tick count retrieved from the specified @p agent.
 *
 * @param[out] system_tick The translated HSA system domain clock counter tick.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p system_tick is NULL;
 */
hsa_status_t HSA_API
    hsa_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent,
                                                    uint64_t agent_tick,
                                                    uint64_t* system_tick);

/**
 * @brief Signal attribute flags.
 */
typedef enum {
  /**
   * Signal will only be consumed by AMD GPUs.  Limits signal consumption to
   * AMD GPU agents only.  Ignored if @p num_consumers is not zero (all agents).
   */
  HSA_AMD_SIGNAL_AMD_GPU_ONLY = 1,
  /**
   * Signal may be used for interprocess communication.
   * IPC signals can be read, written, and waited on from any process.
   * Profiling using an IPC enabled signal is only supported in a single process
   * at a time.  Producing profiling data in one process and consuming it in
   * another process is undefined.
   */
  HSA_AMD_SIGNAL_IPC = 2,
} hsa_amd_signal_attribute_t;

/**
 * @brief Create a signal with specific attributes.
 *
 * @param[in] initial_value Initial value of the signal.
 *
 * @param[in] num_consumers Size of @p consumers. A value of 0 indicates that
 * any agent might wait on the signal.
 *
 * @param[in] consumers List of agents that might consume (wait on) the
 * signal. If @p num_consumers is 0, this argument is ignored; otherwise, the
 * HSA runtime might use the list to optimize the handling of the signal
 * object. If an agent not listed in @p consumers waits on the returned
 * signal, the behavior is undefined. The memory associated with @p consumers
 * can be reused or freed after the function returns.
 *
 * @param[in] attributes Requested signal attributes.  Multiple signal attributes
 * may be requested by combining them with bitwise OR.  Requesting no attributes
 * (@p attributes == 0) results in the same signal as would have been obtained
 * via hsa_signal_create.
 *
 * @param[out] signal Pointer to a memory location where the HSA runtime will
 * store the newly created signal handle. Must not be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime failed to allocate
 * the required resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p signal is NULL, @p
 * num_consumers is greater than 0 but @p consumers is NULL, or @p consumers
 * contains duplicates.
 */
hsa_status_t HSA_API hsa_amd_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
                                           const hsa_agent_t* consumers, uint64_t attributes,
                                           hsa_signal_t* signal);

/**
 * @brief Returns a pointer to the value of a signal.
 *
 * Use of this API does not modify the lifetime of ::signal and any
 * hsa_signal_value_t retrieved by this API has lifetime equal to that of
 * ::signal.
 *
 * This API is intended for partial interoperability with non-HSA compatible
 * devices and should not be used where HSA interfaces are available.
 *
 * Use of the signal value must comply with use restritions of ::signal.
 * Use may result in data races if the operations performed are not platform
 * atomic.  Use with HSA_AMD_SIGNAL_AMD_GPU_ONLY or HSA_AMD_SIGNAL_IPC
 * attributed signals is required.
 *
 * @param[in] Signal handle to extract the signal value pointer from.
 *
 * @param[out] Location where the extracted signal value pointer will be placed.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SIGNAL signal is not a valid hsa_signal_t
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT value_ptr is NULL.
 */
hsa_status_t hsa_amd_signal_value_pointer(hsa_signal_t signal,
                                          volatile hsa_signal_value_t** value_ptr);

/**
 * @brief Asyncronous signal handler function type.
 *
 * @details Type definition of callback function to be used with
 * hsa_amd_signal_async_handler. This callback is invoked if the associated
 * signal and condition are met. The callback receives the value of the signal
 * which satisfied the associated wait condition and a user provided value. If
 * the callback returns true then the callback will be called again if the
 * associated signal and condition are satisfied again. If the callback returns
 * false then it will not be called again.
 *
 * @param[in] value Contains the value of the signal observed by
 * hsa_amd_signal_async_handler which caused the signal handler to be invoked.
 *
 * @param[in] arg Contains the user provided value given when the signal handler
 * was registered with hsa_amd_signal_async_handler
 *
 * @retval true resumes monitoring the signal with this handler (as if calling
 * hsa_amd_signal_async_handler again with identical parameters)
 *
 * @retval false stops monitoring the signal with this handler (handler will
 * not be called again for this signal)
 *
 */
typedef bool (*hsa_amd_signal_handler)(hsa_signal_value_t value, void* arg);

/**
 * @brief Register asynchronous signal handler function.
 *
 * @details Allows registering a callback function and user provided value with
 * a signal and wait condition. The callback will be invoked if the associated
 * signal and wait condition are satisfied. Callbacks will be invoked serially
 * but in an arbitrary order so callbacks should be independent of each other.
 * After being invoked a callback may continue to wait for its associated signal
 * and condition and, possibly, be invoked again. Or the callback may stop
 * waiting. If the callback returns true then it will continue waiting and may
 * be called again. If false then the callback will not wait again and will not
 * be called again for the associated signal and condition. It is possible to
 * register the same callback multiple times with the same or different signals
 * and/or conditions. Each registration of the callback will be treated entirely
 * independently.
 *
 * @param[in] signal hsa signal to be asynchronously monitored
 *
 * @param[in] cond condition value to monitor for
 *
 * @param[in] value signal value used in condition expression
 *
 * @param[in] handler asynchronous signal handler invoked when signal's
 * condition is met
 *
 * @param[in] arg user provided value which is provided to handler when handler
 * is invoked
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SIGNAL signal is not a valid hsa_signal_t
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT handler is invalid (NULL)
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime is out of
 * resources or blocking signals are not supported by the HSA driver component.
 *
 */
hsa_status_t HSA_API
    hsa_amd_signal_async_handler(hsa_signal_t signal,
                                 hsa_signal_condition_t cond,
                                 hsa_signal_value_t value,
                                 hsa_amd_signal_handler handler, void* arg);

/**
 * @brief Call a function asynchronously
 *
 * @details Provides access to the runtime's asynchronous event handling thread
 * for general asynchronous functions.  Functions queued this way are executed
 * in the same manner as if they were a signal handler who's signal is
 * satisfied.
 *
 * @param[in] callback asynchronous function to be invoked
 *
 * @param[in] arg user provided value which is provided to handler when handler
 * is invoked
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT handler is invalid (NULL)
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The HSA runtime is out of
 * resources or blocking signals are not supported by the HSA driver component.
 *
 */
hsa_status_t HSA_API
    hsa_amd_async_function(void (*callback)(void* arg), void* arg);

/**
 * @brief Wait for any signal-condition pair to be satisfied.
 *
 * @details Allows waiting for any of several signal and conditions pairs to be
 * satisfied. The function returns the index into the list of signals of the
 * first satisfying signal-condition pair. The value of the satisfying signal's
 * value is returned in satisfying_value unless satisfying_value is NULL. This
 * function provides only relaxed memory semantics.
 */
uint32_t HSA_API
    hsa_amd_signal_wait_any(uint32_t signal_count, hsa_signal_t* signals,
                            hsa_signal_condition_t* conds,
                            hsa_signal_value_t* values, uint64_t timeout_hint,
                            hsa_wait_state_t wait_hint,
                            hsa_signal_value_t* satisfying_value);

/**
 * @brief Query image limits.
 *
 * @param[in] agent A valid agent.
 *
 * @param[in] attribute HSA image info attribute to query.
 *
 * @param[out] value Pointer to an application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_QUEUE @p value is NULL or @p attribute <
 * HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS or @p attribute >
 * HSA_EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS.
 *
 */
hsa_status_t HSA_API hsa_amd_image_get_info_max_dim(hsa_agent_t agent,
                                                    hsa_agent_info_t attribute,
                                                    void* value);

/**
 * @brief Set a queue's CU affinity mask.
 *
 * @details Enables the queue to run on only selected CUs.  The given mask is
 * combined by bitwise AND with any device wide mask in HSA_CU_MASK before
 * being applied.
 * If num_cu_mask_count is 0 then the request is interpreted as a request to
 * enable all CUs and no cu_mask array need be given.
 *
 * @param[in] queue A pointer to HSA queue.
 *
 * @param[in] num_cu_mask_count Size of CUMask bit array passed in, in bits.
 *
 * @param[in] cu_mask Bit-vector representing the CU mask.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_CU_MASK_REDUCED The function was successfully executed
 * but the given mask attempted to enable a CU which was disabled by
 * HSA_CU_MASK.  CUs disabled by HSA_CU_MASK remain disabled.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_QUEUE @p queue is NULL or invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p num_cu_mask_count is not
 * a multiple of 32 or @p num_cu_mask_count is not 0 and cu_mask is NULL.
 * Devices with work group processors must even-index contiguous pairwise
 * CU enable e.g. 0x33(b'110011) is valid while 0x5(0x101) and 0x6(b'0110)
 * are invalid.
 *
 */
hsa_status_t HSA_API hsa_amd_queue_cu_set_mask(const hsa_queue_t* queue,
                                               uint32_t num_cu_mask_count,
                                               const uint32_t* cu_mask);

/**
 * @brief Retrieve a queue's CU affinity mask.
 *
 * @details Returns the first num_cu_mask_count bits of a queue's CU mask.
 * Ensure that num_cu_mask_count is at least as large as
 * HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT to retrieve the entire mask.
 *
 * @param[in] queue A pointer to HSA queue.
 *
 * @param[in] num_cu_mask_count Size of CUMask bit array passed in, in bits.
 *
 * @param[out] cu_mask Bit-vector representing the CU mask.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_QUEUE @p queue is NULL or invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p num_cu_mask_count is 0, not
 * a multiple of 32 or @p cu_mask is NULL.
 *
 */
hsa_status_t HSA_API hsa_amd_queue_cu_get_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count,
                                               uint32_t* cu_mask);

/**
 * @brief Memory segments associated with a memory pool.
 */
typedef enum {
  /**
   * Global segment. Used to hold data that is shared by all agents.
   */
  HSA_AMD_SEGMENT_GLOBAL = 0,
  /**
   * Read-only segment. Used to hold data that remains constant during the
   * execution of a kernel.
   */
  HSA_AMD_SEGMENT_READONLY = 1,
  /**
   * Private segment. Used to hold data that is local to a single work-item.
   */
  HSA_AMD_SEGMENT_PRIVATE = 2,
  /**
   * Group segment. Used to hold data that is shared by the work-items of a
   * work-group.
   */
  HSA_AMD_SEGMENT_GROUP = 3,
} hsa_amd_segment_t;

/**
 * @brief A memory pool encapsulates physical storage on an agent
 * along with a memory access model.
 *
 * @details A memory pool encapsulates a physical partition of an agent's
 * memory system along with a memory access model.  Division of a single
 * memory system into separate pools allows querying each partition's access
 * path properties (see ::hsa_amd_agent_memory_pool_get_info). Allocations
 * from a pool are preferentially bound to that pool's physical partition.
 * Binding to the pool's preferential physical partition may not be
 * possible or persistent depending on the system's memory policy
 * and/or state which is beyond the scope of HSA APIs.
 *
 * For example, a multi-node NUMA memory system may be represented by multiple
 * pool's with each pool providing size and access path information for the
 * partition it represents.  Allocations from a pool are preferentially bound
 * to the pool's partition (which in this example is a NUMA node) while
 * following its memory access model. The actual placement may vary or migrate
 * due to the system's NUMA policy and state, which is beyond the scope of
 * HSA APIs.
 */ 
typedef struct hsa_amd_memory_pool_s {
  /**
   * Opaque handle.
   */
  uint64_t handle;
} hsa_amd_memory_pool_t;

typedef enum hsa_amd_memory_pool_global_flag_s {
  /**
   * The application can use allocations in the memory pool to store kernel
   * arguments, and provide the values for the kernarg segment of
   * a kernel dispatch.
   */
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT = 1,
  /**
   * Updates to memory in this pool conform to HSA memory consistency model.
   * If this flag is set, then ::HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED
   * must not be set.
   */
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED = 2,
  /**
   * Writes to memory in this pool can be performed by a single agent at a time.
   */
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4,

  /** Updates to memory in this memory pool have extended scope, acting as
   * system-scope atomics for variables in memory regions of this type.
   * Note: On non-compliant systems, device-specific actions may be required
   * for system-scope coherence. */
  HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED = 8,

} hsa_amd_memory_pool_global_flag_t;

typedef enum hsa_amd_memory_pool_location_s {
    /**
     * This memory pool resides on the host (CPU)
     */
    HSA_AMD_MEMORY_POOL_LOCATION_CPU = 0,
    /**
     * This memory pool resides on a GPU
     */
    HSA_AMD_MEMORY_POOL_LOCATION_GPU = 1
} hsa_amd_memory_pool_location_t;

/**
 * @brief Memory pool features.
 */
typedef enum {
  /**
  * Segment where the memory pool resides. The type of this attribute is
  * ::hsa_amd_segment_t.
  */
  HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0,
  /**
  * Flag mask. The value of this attribute is undefined if the value of
  * ::HSA_AMD_MEMORY_POOL_INFO_SEGMENT is not ::HSA_AMD_SEGMENT_GLOBAL. The type
  * of
  * this attribute is uint32_t, a bit-field of
  * ::hsa_amd_memory_pool_global_flag_t
  * values.
  */
  HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1,
  /**
  * Size of this pool, in bytes. The type of this attribute is size_t.
  */
  HSA_AMD_MEMORY_POOL_INFO_SIZE = 2,
  /**
  * Indicates whether memory in this pool can be allocated using
  * ::hsa_amd_memory_pool_allocate. The type of this attribute is bool.
  *
  * The value of this flag is always false for memory pools in the group and
  * private segments.
  */
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED = 5,
  /**
   * Allocation granularity of buffers allocated by
   * ::hsa_amd_memory_pool_allocate
   * in this memory pool. The size of a buffer allocated in this pool is a
   * multiple of the value of this attribute. While this is the minimum size of
   * allocation allowed, it is recommened to use
   * HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE to obtain the recommended
   * allocation granularity size for this pool.
   * The value of this attribute is only defined if
   * ::HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED is true for
   * this pool. The type of this attribute is size_t.
   */
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6,
  /**
   * Alignment of buffers allocated by ::hsa_amd_memory_pool_allocate in this
   * pool. The value of this attribute is only defined if
   * ::HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED is true for this pool, and
   * must be a power of 2. The type of this attribute is size_t.
   */
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT = 7,
  /**
   * This memory_pool can be made directly accessible by all the agents in the
   * system (::hsa_amd_agent_memory_pool_get_info does not return
   * ::HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED for any agent). The type of this
   * attribute is bool.
   */
  HSA_AMD_MEMORY_POOL_INFO_ACCESSIBLE_BY_ALL = 15,
  /**
   * Maximum aggregate allocation size in bytes. The type of this attribute
   * is size_t.
   */
  HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE = 16,
  /**
   * Location of this memory pool. The type of this attribute
   * is hsa_amd_memory_pool_location_t.
   */
  HSA_AMD_MEMORY_POOL_INFO_LOCATION = 17,
  /**
   * Internal block size for allocations. This would also be the recommended
   * granularity size for allocations as this prevents internal fragmentation.
   * The value of this attribute is only defined if
   * ::HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED is true for this pool.
   * The size of this attribute is size_t.
   */
  HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = 18,
} hsa_amd_memory_pool_info_t;

/**
 * @brief Memory pool flag used to specify allocation directives
 *
 */
typedef enum hsa_amd_memory_pool_flag_s {
  /**
   * Allocates memory that conforms to standard HSA memory consistency model
   */
  HSA_AMD_MEMORY_POOL_STANDARD_FLAG = 0,
  /**
   * Allocates fine grain memory type where memory ordering is per point to point
   * connection. Atomic memory operations on these memory buffers are not
   * guaranteed to be visible at system scope.
   */
  HSA_AMD_MEMORY_POOL_PCIE_FLAG = (1 << 0),
  /**
   *  Allocates physically contiguous memory
   */
  HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG = (1 << 1),

} hsa_amd_memory_pool_flag_t;

/**
 * @brief Get the current value of an attribute of a memory pool.
 *
 * @param[in] memory_pool A valid memory pool.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to a application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 */
hsa_status_t HSA_API
    hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                 hsa_amd_memory_pool_info_t attribute,
                                 void* value);

/**
 * @brief Iterate over the memory pools associated with a given agent, and
 * invoke an application-defined callback on every iteration.
 *
 * @details An agent can directly access buffers located in some memory pool, or
 * be enabled to access them by the application (see ::hsa_amd_agents_allow_access),
 * yet that memory pool may not be returned by this function for that given
 * agent.
 *
 * A memory pool of fine-grained type must be associated only with the host.
 *
 * @param[in] agent A valid agent.
 *
 * @param[in] callback Callback to be invoked on the same thread that called
 * ::hsa_amd_agent_iterate_memory_pools, serially, once per memory pool that is
 * associated with the agent.  The HSA runtime passes two arguments to the
 * callback: the memory pool, and the application data.  If @p callback
 * returns a status other than ::HSA_STATUS_SUCCESS for a particular iteration,
 * the traversal stops and ::hsa_amd_agent_iterate_memory_pools returns that status
 * value.
 *
 * @param[in] data Application data that is passed to @p callback on every
 * iteration. May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL.
 */
hsa_status_t HSA_API hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
    void* data);

/**
 * @brief Allocate a block of memory (or buffer) in the specified pool.
 *
 * @param[in] memory_pool Memory pool where to allocate memory from. The memory
 * pool must have the ::HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED flag set.
 *
 * @param[in] size Allocation size, in bytes. Must not be zero. This value is
 * rounded up to the nearest multiple of
 * ::HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE in @p memory_pool.
 *
 * @param[in] flags A bit-field that is used to specify allocation
 * directives.
 *
 * @param[out] ptr Pointer to the location where to store the base virtual
 * address of
 * the allocated block. The returned base address is aligned to the value of
 * ::HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT in @p memory_pool. If the
 * allocation fails, the returned value is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES No memory is available.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_MEMORY_POOL The memory pool is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION The host is not allowed to
 * allocate memory in @p memory_pool, or @p size is greater than
 * the value of HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE in @p memory_pool.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p ptr is NULL, or @p size is 0,
 * or flags is not 0.
 *
 */
hsa_status_t HSA_API
    hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                 uint32_t flags, void** ptr);

/**
 * @brief Deallocate a block of memory previously allocated using
 * ::hsa_amd_memory_pool_allocate.
 *
 * @param[in] ptr Pointer to a memory block. If @p ptr does not match a value
 * previously returned by ::hsa_amd_memory_pool_allocate, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 */
hsa_status_t HSA_API hsa_amd_memory_pool_free(void* ptr);

/**
 * @brief Asynchronously copy a block of memory from the location pointed to by
 * @p src on the @p src_agent to the memory block pointed to by @p dst on the @p
 * dst_agent.
 * Because the DMA engines used may not be in the same coherency domain, the caller must ensure
 * that buffers are system-level coherent. In general this requires the sending device to have
 * released the buffer to system scope prior to executing the copy API and the receiving device
 * must execute a system scope acquire fence prior to use of the destination buffer.
 *
 * @param[out] dst Buffer where the content is to be copied.
 *
 * @param[in] dst_agent Agent associated with the @p dst. The agent must be able to directly
 * access both the source and destination buffers in their current locations.
 * May be zero in which case the runtime will attempt to discover the destination agent.
 * Discovery may have variable and/or high latency.
 *
 * @param[in] src A valid pointer to the source of data to be copied. The source
 * buffer must not overlap with the destination buffer, otherwise the copy will succeed
 * but contents of @p dst is undefined.
 *
 * @param[in] src_agent Agent associated with the @p src. The agent must be able to directly
 * access both the source and destination buffers in their current locations.
 * May be zero in which case the runtime will attempt to discover the destination agent.
 * Discovery may have variable and/or high latency.
 *
 * @param[in] size Number of bytes to copy. If @p size is 0, no copy is
 * performed and the function returns success. Copying a number of bytes larger
 * than the size of the buffers pointed by @p dst or @p src results in undefined
 * behavior.
 *
 * @param[in] num_dep_signals Number of dependent signals. Can be 0.
 *
 * @param[in] dep_signals List of signals that must be waited on before the copy
 * operation starts. The copy will start after every signal has been observed with
 * the value 0. The dependent signal should not include completion signal from
 * hsa_amd_memory_async_copy operation to be issued in future as that can result
 * in a deadlock. If @p num_dep_signals is 0, this argument is ignored.
 *
 * @param[in] completion_signal Signal used to indicate completion of the copy
 * operation. When the copy operation is finished, the value of the signal is
 * decremented. The runtime indicates that an error has occurred during the copy
 * operation by setting the value of the completion signal to a negative
 * number. The signal handle must not be 0.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully. The
 * application is responsible for checking for asynchronous error conditions
 * (see the description of @p completion_signal).
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT An agent is invalid or no discovered agent has access.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_SIGNAL @p completion_signal is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT The source or destination
 * pointers are NULL, or the completion signal is 0.
 */
hsa_status_t HSA_API
    hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent, const void* src,
                              hsa_agent_t src_agent, size_t size,
                              uint32_t num_dep_signals,
                              const hsa_signal_t* dep_signals,
                              hsa_signal_t completion_signal);

/**
 * @brief Asynchronously copy a block of memory from the location pointed to by
 * @p src on the @p src_agent to the memory block pointed to by @p dst on the @p
 * dst_agent on engine_id.
 *
 * WARNING: Concurrent use of this call with hsa_amd_memory_async_copy can result
 * in resource conflicts as HSA runtime will auto assign engines with the latter
 * call.  Approach using both calls concurrently with caution.
 *
 * All param definitions are identical to hsa_amd_memory_async_copy with the
 * exception of engine_id and force_copy_on_sdma.
 *
 * @param[in] - engine_id Target engine defined by hsa_amd_sdma_engine_id_t.
 * Client should use hsa_amd_memory_copy_engine_status first to get the ID
 * availability.
 *
 * @param[in] - force_copy_on_sdma By default, blit kernel copies are used when
 * dst_agent == src_agent.  Setting this to true will force the copy over SDMA1.
 *
 * All return definitions are identical to hsa_amd_memory_async_copy with the
 * following ammendments:
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT The source or destination
 * pointers are NULL, or the completion signal is 0 or engine_id is improperly
 * bounded.
 */
hsa_status_t HSA_API
    hsa_amd_memory_async_copy_on_engine(void* dst, hsa_agent_t dst_agent, const void* src,
                              hsa_agent_t src_agent, size_t size,
                              uint32_t num_dep_signals,
                              const hsa_signal_t* dep_signals,
                              hsa_signal_t completion_signal,
                              hsa_amd_sdma_engine_id_t engine_id,
                              bool force_copy_on_sdma);
/**
 * @brief Reports the availability of SDMA copy engines.
 *
 * @param[in] dst_agent Destination agent of copy status direction.
 *
 * @param[in] src_agent Source agent of copy status direction.
 *
 * @param[out] engine_ids_mask returns available SDMA engine IDs that can be masked
 * with hsa_amd_sdma_engine_id_t.
 *
 * @retval ::HSA_STATUS_SUCCESS Agent has available SDMA engines.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Agent does not have available SDMA engines.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT dst_agent and src_agent are the same as
 * dst_agent == src_agent is generally used for shader copies.
 */
hsa_status_t HSA_API
    hsa_amd_memory_copy_engine_status(hsa_agent_t dst_agent, hsa_agent_t src_agent,
                                      uint32_t *engine_ids_mask);

/*
[Provisional API]
Pitched memory descriptor.
All elements must be 4 byte aligned.  Pitch and slice are in bytes.
*/
typedef struct hsa_pitched_ptr_s {
  void* base;
  size_t pitch;
  size_t slice;
} hsa_pitched_ptr_t;

/*
[Provisional API]
Copy direction flag.
*/
typedef enum {
  hsaHostToHost = 0,
  hsaHostToDevice = 1,
  hsaDeviceToHost = 2,
  hsaDeviceToDevice = 3
} hsa_amd_copy_direction_t;

/*
[Provisional API]
SDMA 3D memory copy API.  The same requirements must be met by src and dst as in
hsa_amd_memory_async_copy.
Both src and dst must be directly accessible to the copy_agent during the copy, src and dst rects
must not overlap.
CPU agents are not supported.  API requires SDMA and will return an error if SDMA is not available.
Offsets and range carry x in bytes, y and z in rows and layers.
*/
hsa_status_t HSA_API hsa_amd_memory_async_copy_rect(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal);

/**
 * @brief Type of accesses to a memory pool from a given agent.
 */
typedef enum {
  /**
  * The agent cannot directly access any buffer in the memory pool.
  */
  HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED = 0,
  /**
  * The agent can directly access a buffer located in the pool; the application
  * does not need to invoke ::hsa_amd_agents_allow_access.
  */
  HSA_AMD_MEMORY_POOL_ACCESS_ALLOWED_BY_DEFAULT = 1,
  /**
  * The agent can directly access a buffer located in the pool, but only if the
  * application has previously requested access to that buffer using
  * ::hsa_amd_agents_allow_access.
  */
  HSA_AMD_MEMORY_POOL_ACCESS_DISALLOWED_BY_DEFAULT = 2
} hsa_amd_memory_pool_access_t;

/**
 * @brief Properties of the relationship between an agent a memory pool.
 */
typedef enum {
  /**
  * Hyper-transport bus type.
  */
  HSA_AMD_LINK_INFO_TYPE_HYPERTRANSPORT = 0,

  /**
  * QPI bus type.
  */
  HSA_AMD_LINK_INFO_TYPE_QPI = 1,

  /**
  * PCIe bus type.
  */
  HSA_AMD_LINK_INFO_TYPE_PCIE = 2,

  /**
  * Infiniband bus type.
  */
  HSA_AMD_LINK_INFO_TYPE_INFINBAND = 3,

  /**
  * xGMI link type.
  */
  HSA_AMD_LINK_INFO_TYPE_XGMI = 4

} hsa_amd_link_info_type_t;

/**
 * @brief Link properties when accessing the memory pool from the specified
 * agent.
 */
typedef struct hsa_amd_memory_pool_link_info_s {
  /**
  * Minimum transfer latency (rounded to ns).
  */
  uint32_t min_latency;

  /**
  * Maximum transfer latency (rounded to ns).
  */
  uint32_t max_latency;

  /**
  * Minimum link interface bandwidth in MB/s.
  */
  uint32_t min_bandwidth;

  /**
  * Maximum link interface bandwidth in MB/s.
  */
  uint32_t max_bandwidth;

  /**
  * Support for 32-bit atomic transactions.
  */
  bool atomic_support_32bit;

  /**
  * Support for 64-bit atomic transactions.
  */
  bool atomic_support_64bit;

  /**
  * Support for cache coherent transactions.
  */
  bool coherent_support;

  /**
  * The type of bus/link.
  */
  hsa_amd_link_info_type_t link_type;

  /**
   * NUMA distance of memory pool relative to querying agent
   */
  uint32_t numa_distance;
} hsa_amd_memory_pool_link_info_t;

/**
 * @brief Properties of the relationship between an agent a memory pool.
 */
typedef enum {
  /**
  * Access to buffers located in the memory pool. The type of this attribute
  * is ::hsa_amd_memory_pool_access_t.
  *
  * An agent can always directly access buffers currently located in a memory
  * pool that is associated (the memory_pool is one of the values returned by
  * ::hsa_amd_agent_iterate_memory_pools on the agent) with that agent. If the
  * buffer is currently located in a memory pool that is not associated with
  * the agent, and the value returned by this function for the given
  * combination of agent and memory pool is not
  * HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED, the application still needs to invoke
  * ::hsa_amd_agents_allow_access in order to gain direct access to the buffer.
  *
  * If the given agent can directly access buffers the pool, the result is not
  * HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED. If the memory pool is associated with
  * the agent, or it is of fined-grained type, the result must not be
  * HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED. If the memory pool is not associated
  * with the agent, and does not reside in the global segment, the result must
  * be HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED.
  */
  HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS = 0,

  /**
  * Number of links to hop when accessing the memory pool from the specified
  * agent. The value of this attribute is zero if the memory pool is associated
  * with the agent, or if the access type is
  * HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED. The type of this attribute is
  * uint32_t.
  */
  HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS = 1,

  /**
  * Details of each link hop when accessing the memory pool starting from the
  * specified agent. The type of this attribute is an array size of
  * HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS with each element containing
  * ::hsa_amd_memory_pool_link_info_t.
  */
  HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO = 2

} hsa_amd_agent_memory_pool_info_t;

/**
 * @brief Get the current value of an attribute of the relationship between an
 * agent and a memory pool.
 *
 * @param[in] agent Agent.
 *
 * @param[in] memory_pool Memory pool.
 *
 * @param[in] attribute Attribute to query.
 *
 * @param[out] value Pointer to a application-allocated buffer where to store
 * the value of the attribute. If the buffer passed by the application is not
 * large enough to hold the value of @p attribute, the behavior is undefined.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 */
hsa_status_t HSA_API hsa_amd_agent_memory_pool_get_info(
    hsa_agent_t agent, hsa_amd_memory_pool_t memory_pool,
    hsa_amd_agent_memory_pool_info_t attribute, void* value);

/**
 * @brief Enable direct access to a buffer from a given set of agents.
 *
 * @details
 *
 * Upon return, only the listed agents and the agent associated with the
 * buffer's memory pool have direct access to the @p ptr.
 *
 * Any agent that has access to the buffer before and after the call to
 * ::hsa_amd_agents_allow_access will also have access while
 * ::hsa_amd_agents_allow_access is in progress.
 *
 * The caller is responsible for ensuring that each agent in the list
 * must be able to access the memory pool containing @p ptr
 * (using ::hsa_amd_agent_memory_pool_get_info with ::HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS attribute),
 * otherwise error code is returned.
 *
 * @param[in] num_agents Size of @p agents.
 *
 * @param[in] agents List of agents. If @p num_agents is 0, this argument is
 * ignored.
 *
 * @param[in] flags A list of bit-field that is used to specify access
 * information in a per-agent basis. This is currently reserved and must be NULL.
 *
 * @param[in] ptr A buffer previously allocated using ::hsa_amd_memory_pool_allocate.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p num_agents is 0, or @p agents
 * is NULL, @p flags is not NULL, or attempting to enable access to agent(s)
 * because @p ptr is allocated from an inaccessible pool.
 *
 */
hsa_status_t HSA_API
    hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents,
                                const uint32_t* flags, const void* ptr);

/**
 * @brief Query if buffers currently located in some memory pool can be
 * relocated to a destination memory pool.
 *
 * @details If the returned value is non-zero, a migration of a buffer to @p
 * dst_memory_pool using ::hsa_amd_memory_migrate may nevertheless fail due to
 * resource limitations.
 *
 * @param[in] src_memory_pool Source memory pool.
 *
 * @param[in] dst_memory_pool Destination memory pool.
 *
 * @param[out] result Pointer to a memory location where the result of the query
 * is stored. Must not be NULL. If buffers currently located in @p
 * src_memory_pool can be relocated to @p dst_memory_pool, the result is
 * true.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_MEMORY_POOL One of the memory pools is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p result is NULL.
 */
hsa_status_t HSA_API
    hsa_amd_memory_pool_can_migrate(hsa_amd_memory_pool_t src_memory_pool,
                                    hsa_amd_memory_pool_t dst_memory_pool,
                                    bool* result);

/**
 * @brief Relocate a buffer to a new memory pool.
 *
 * @details When a buffer is migrated, its virtual address remains the same but
 * its physical contents are moved to the indicated memory pool.
 *
 * After migration, only the agent associated with the destination pool will have access.
 *
 * The caller is also responsible for ensuring that the allocation in the
 * source memory pool where the buffer is currently located can be migrated to the
 * specified destination memory pool (using ::hsa_amd_memory_pool_can_migrate returns a value of true
 * for the source and destination memory pools), otherwise behavior is undefined.
 *
 * The caller must ensure that the buffer is not accessed while it is migrated.
 *
 * @param[in] ptr Buffer to be relocated. The buffer must have been released to system
 * prior to call this API.  The buffer will be released to system upon completion.
 *
 * @param[in] memory_pool Memory pool where to place the buffer.
 *
 * @param[in] flags A bit-field that is used to specify migration
 * information. Must be zero.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_MEMORY_POOL The destination memory pool is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES There is a failure in
 * allocating the necessary resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p flags is not 0.
 */
hsa_status_t HSA_API hsa_amd_memory_migrate(const void* ptr,
                                            hsa_amd_memory_pool_t memory_pool,
                                            uint32_t flags);

/**
 *
 * @brief Pin a host pointer allocated by C/C++ or OS allocator (i.e. ordinary system DRAM) and
 * return a new pointer accessible by the @p agents. If the @p host_ptr overlaps with previously
 * locked memory, then the overlap area is kept locked (i.e multiple mappings are permitted). In
 * this case, the same input @p host_ptr may give different locked @p agent_ptr and when it does,
 * they are not necessarily coherent (i.e. accessing either @p agent_ptr is not equivalent).
 * Accesses to @p agent_ptr are coarse grained.
 *
 * @param[in] host_ptr A buffer allocated by C/C++ or OS allocator.
 *
 * @param[in] size The size to be locked.
 *
 * @param[in] agents Array of agent handle to gain access to the @p host_ptr.
 * If this parameter is NULL and the @p num_agent is 0, all agents
 * in the platform will gain access to the @p host_ptr.
 *
 * @param[out] agent_ptr Pointer to the location where to store the new address.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES There is a failure in
 * allocating the necessary resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT One or more agent in @p agents is
 * invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p size is 0 or @p host_ptr or
 * @p agent_ptr is NULL or @p agents not NULL but @p num_agent is 0 or @p agents
 * is NULL but @p num_agent is not 0.
 */
hsa_status_t HSA_API hsa_amd_memory_lock(void* host_ptr, size_t size,
                                         hsa_agent_t* agents, int num_agent,
                                         void** agent_ptr);

/**
 *
 * @brief Pin a host pointer allocated by C/C++ or OS allocator (i.e. ordinary system DRAM) and
 * return a new pointer accessible by the @p agents. If the @p host_ptr overlaps with previously
 * locked memory, then the overlap area is kept locked (i.e. multiple mappings are permitted).
 * In this case, the same input @p host_ptr may give different locked @p agent_ptr and when it
 * does, they are not necessarily coherent (i.e. accessing either @p agent_ptr is not equivalent).
 * Acesses to the memory via @p agent_ptr have the same access properties as memory allocated from
 * @p pool as determined by ::hsa_amd_memory_pool_get_info and ::hsa_amd_agent_memory_pool_get_info
 * (ex. coarse/fine grain, platform atomic support, link info).  Physical composition and placement
 * of the memory (ex. page size, NUMA binding) is not changed.
 *
 * @param[in] host_ptr A buffer allocated by C/C++ or OS allocator.
 *
 * @param[in] size The size to be locked.
 *
 * @param[in] agents Array of agent handle to gain access to the @p host_ptr.
 * If this parameter is NULL and the @p num_agent is 0, all agents
 * in the platform will gain access to the @p host_ptr.
 *
 * @param[in] pool Global memory pool owned by a CPU agent.
 *
 * @param[in] flags A bit-field that is used to specify allocation
 * directives. Reserved parameter, must be 0.
 *
 * @param[out] agent_ptr Pointer to the location where to store the new address.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES There is a failure in
 * allocating the necessary resources.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT One or more agent in @p agents is
 * invalid or can not access @p pool.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_MEMORY_POOL @p pool is invalid or not owned
 * by a CPU agent.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p size is 0 or @p host_ptr or
 * @p agent_ptr is NULL or @p agents not NULL but @p num_agent is 0 or @p agents
 * is NULL but @p num_agent is not 0 or flags is not 0.
 */
hsa_status_t HSA_API hsa_amd_memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents,
                                                 int num_agent, hsa_amd_memory_pool_t pool,
                                                 uint32_t flags, void** agent_ptr);

/**
 *
 * @brief Unpin the host pointer previously pinned via ::hsa_amd_memory_lock or
 * ::hsa_amd_memory_lock_to_pool.
 *
 * @details The behavior is undefined if the host pointer being unpinned does not
 * match previous pinned address or if the host pointer was already deallocated.
 *
 * @param[in] host_ptr A buffer allocated by C/C++ or OS allocator that was
 * pinned previously via ::hsa_amd_memory_lock or ::hsa_amd_memory_lock_to_pool.
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 */
hsa_status_t HSA_API hsa_amd_memory_unlock(void* host_ptr);

/**
 * @brief Sets the first @p count of uint32_t of the block of memory pointed by
 * @p ptr to the specified @p value.
 *
 * @param[in] ptr Pointer to the block of memory to fill.
 *
 * @param[in] value Value to be set.
 *
 * @param[in] count Number of uint32_t element to be set to the value.
 *
 * @retval HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p ptr is NULL or
 * not 4 bytes aligned
 *
 * @retval HSA_STATUS_ERROR_INVALID_ALLOCATION if the given memory
 * region was not allocated with HSA runtime APIs.
 *
 */
hsa_status_t HSA_API
    hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count);

/**
 * @brief Maps an interop object into the HSA flat address space and establishes
 * memory residency.  The metadata pointer is valid during the lifetime of the
 * map (until hsa_amd_interop_unmap_buffer is called).
 * Multiple calls to hsa_amd_interop_map_buffer with the same interop_handle
 * result in multiple mappings with potentially different addresses and
 * different metadata pointers.  Concurrent operations on these addresses are
 * not coherent.  Memory must be fenced to system scope to ensure consistency,
 * between mappings and with any views of this buffer in the originating
 * software stack.
 *
 * @param[in] num_agents Number of agents which require access to the memory
 *
 * @param[in] agents List of accessing agents.
 *
 * @param[in] interop_handle Handle of interop buffer (dmabuf handle in Linux)
 *
 * @param [in] flags Reserved, must be 0
 *
 * @param[out] size Size in bytes of the mapped object
 *
 * @param[out] ptr Base address of the mapped object
 *
 * @param[out] metadata_size Size of metadata in bytes, may be NULL
 *
 * @param[out] metadata Pointer to metadata, may be NULL
 *
 * @retval HSA_STATUS_SUCCESS if successfully mapped
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT all other errors
 */
hsa_status_t HSA_API hsa_amd_interop_map_buffer(uint32_t num_agents,
                                        hsa_agent_t* agents,
                                        int interop_handle,
                                        uint32_t flags,
                                        size_t* size,
                                        void** ptr,
                                        size_t* metadata_size,
                                        const void** metadata);

/**
 * @brief Removes a previously mapped interop object from HSA's flat address space.
 * Ends lifetime for the mapping's associated metadata pointer.
 */
hsa_status_t HSA_API hsa_amd_interop_unmap_buffer(void* ptr);

/**
 * @brief Encodes an opaque vendor specific image format.  The length of data
 * depends on the underlying format.  This structure must not be copied as its
 * true length can not be determined.
 */
typedef struct hsa_amd_image_descriptor_s {
  /*
  Version number of the descriptor
  */
  uint32_t version;

  /*
  Vendor and device PCI IDs for the format as VENDOR_ID<<16|DEVICE_ID.
  */
  uint32_t deviceID;

  /*
  Start of vendor specific data.
  */
  uint32_t data[1];
} hsa_amd_image_descriptor_t;

/**
 * @brief Creates an image from an opaque vendor specific image format.
 * Does not modify data at image_data.  Intended initially for
 * accessing interop images.
 *
 * @param agent[in] Agent on which to create the image
 *
 * @param[in] image_descriptor[in] Vendor specific image format
 *
 * @param[in] image_data Pointer to image backing store
 *
 * @param[in] access_permission Access permissions for the image object
 *
 * @param[out] image Created image object.
 *
 * @retval HSA_STATUS_SUCCESS Image created successfully
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT Bad or mismatched descriptor,
 * null image_data, or mismatched access_permission.
 */
hsa_status_t HSA_API hsa_amd_image_create(
    hsa_agent_t agent,
    const hsa_ext_image_descriptor_t *image_descriptor,
    const hsa_amd_image_descriptor_t *image_layout,
    const void *image_data,
    hsa_access_permission_t access_permission,
    hsa_ext_image_t *image
);

/**
 * @brief Denotes the type of memory in a pointer info query.
 */
typedef enum {
  /*
  Memory is not known to the HSA driver.  Unallocated or unlocked system memory.
  */
  HSA_EXT_POINTER_TYPE_UNKNOWN = 0,
  /*
  Memory was allocated with an HSA memory allocator.
  */
  HSA_EXT_POINTER_TYPE_HSA = 1,
  /*
  System memory which has been locked for use with an HSA agent.

  Memory of this type is normal malloc'd memory and is always accessible to
  the CPU.  Pointer info queries may not include CPU agents in the accessible
  agents list as the CPU has implicit access.
  */
  HSA_EXT_POINTER_TYPE_LOCKED = 2,
  /*
  Memory originated in a graphics component and is shared with ROCr.
  */
  HSA_EXT_POINTER_TYPE_GRAPHICS = 3,
  /*
  Memory has been shared with the local process via ROCr IPC APIs.
  */
  HSA_EXT_POINTER_TYPE_IPC = 4
} hsa_amd_pointer_type_t;

/**
 * @brief Describes a memory allocation known to ROCr.
 * Within a ROCr major version this structure can only grow.
 */
typedef struct hsa_amd_pointer_info_s {
  /*
  Size in bytes of this structure.  Used for version control within a major ROCr
  revision.  Set to sizeof(hsa_amd_pointer_t) prior to calling
  hsa_amd_pointer_info.  If the runtime supports an older version of pointer
  info then size will be smaller on return.  Members starting after the return
  value of size will not be updated by hsa_amd_pointer_info.
  */
  uint32_t size;
  /*
  The type of allocation referenced.
  */
  hsa_amd_pointer_type_t type;
  /*
  Base address at which non-host agents may access the allocation. This field is
  not meaningful if the type of the allocation is HSA_EXT_POINTER_TYPE_UNKNOWN.
  */
  void* agentBaseAddress;
  /*
  Base address at which the host agent may access the allocation. This field is
  not meaningful if the type of the allocation is HSA_EXT_POINTER_TYPE_UNKNOWN.
  */
  void* hostBaseAddress;
  /*
  Size of the allocation. This field is not meaningful if the type of the allocation
  is HSA_EXT_POINTER_TYPE_UNKNOWN.
  */
  size_t sizeInBytes;
  /*
  Application provided value. This field is not meaningful if the type of the
  allocation is HSA_EXT_POINTER_TYPE_UNKNOWN.
  */
  void* userData;
  /*
  Reports an agent which "owns" (ie has preferred access to) the pool in which the
  allocation was
  made.  When multiple agents share equal access to a pool (ex: multiple CPU agents, or multi-die
  GPU boards) any such agent may be returned. This field is not meaningful if
  the type of the allocation is HSA_EXT_POINTER_TYPE_UNKNOWN or if this agent is not available in
  this process, for e.g if this agent is masked using ROCR_VISIBLE_DEVICES.
  */
  hsa_agent_t agentOwner;
  /*
  Contains a bitfield of hsa_amd_memory_pool_global_flag_t values.
  Reports the effective global flags bitmask for the allocation.  This field is not
  meaningful if the type of the allocation is HSA_EXT_POINTER_TYPE_UNKNOWN.
  */
  uint32_t global_flags;
} hsa_amd_pointer_info_t;

/**
 * @brief Retrieves information about the allocation referenced by the given
 * pointer.  Optionally returns the number and list of agents which can
 * directly access the allocation. In case this virtual address is unknown, the
 * pointer type returned will be HSA_EXT_POINTER_TYPE_UNKNOWN and the only fields
 * that are valid after hsa_amd_pointer_info returns are size and type.
 *
 * @param[in] ptr Pointer which references the allocation to retrieve info for.
 *
 * @param[in, out] info Pointer to structure to be filled with allocation info.
 * Data member size must be set to the size of the structure prior to calling
 * hsa_amd_pointer_info.  On return size will be set to the size of the
 * pointer info structure supported by the runtime, if smaller.  Members
 * beyond the returned value of size will not be updated by the API.
 * Must not be NULL.
 *
 * @param[in] alloc Function pointer to an allocator used to allocate the
 * @p accessible array.  If NULL @p accessible will not be returned.
 *
 * @param[out] num_agents_accessible Recieves the count of agents in
 * @p accessible.  If NULL @p accessible will not be returned.
 *
 * @param[out] accessible Recieves a pointer to the array, allocated by @p alloc,
 * holding the list of agents which may directly access the allocation.
 * May be NULL.
 *
 * @retval HSA_STATUS_SUCCESS Info retrieved successfully
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT NULL in @p ptr or @p info.
 */
hsa_status_t HSA_API hsa_amd_pointer_info(const void* ptr,
                                          hsa_amd_pointer_info_t* info,
                                          void* (*alloc)(size_t),
                                          uint32_t* num_agents_accessible,
                                          hsa_agent_t** accessible);

/**
 * @brief Associates an arbitrary pointer with an allocation known to ROCr.
 * The pointer can be fetched by hsa_amd_pointer_info in the userData field.
 *
 * @param[in] ptr Pointer to the first byte of an allocation known to ROCr
 * with which to associate @p userdata.
 *
 * @param[in] userdata Abitrary pointer to associate with the allocation.
 *
 * @retval HSA_STATUS_SUCCESS @p userdata successfully stored.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p ptr is not known to ROCr.
 */
hsa_status_t HSA_API hsa_amd_pointer_info_set_userdata(const void* ptr,
                                                       void* userdata);

/**
 * @brief 256-bit process independent identifier for a ROCr shared memory
 * allocation.
 */
typedef struct hsa_amd_ipc_memory_s {
  uint32_t handle[8];
} hsa_amd_ipc_memory_t;

/**
 * @brief Prepares an allocation for interprocess sharing and creates a
 * handle of type hsa_amd_ipc_memory_t uniquely identifying the allocation.  A
 * handle is valid while the allocation it references remains accessible in
 * any process.  In general applications should confirm that a shared memory
 * region has been attached (via hsa_amd_ipc_memory_attach) in the remote
 * process prior to releasing that memory in the local process.
 * Repeated calls for the same allocation may, but are not required to, return
 * unique handles. The allocation needs to be on memory on an agent of type
 * HSA_DEVICE_TYPE_GPU.
 *
 * @param[in] ptr Pointer to device memory allocated via ROCr APIs to prepare for
 * sharing.
 *
 * @param[in] len Length in bytes of the allocation to share.
 *
 * @param[out] handle Process independent identifier referencing the shared
 * allocation.
 *
 * @retval HSA_STATUS_SUCCESS allocation is prepared for interprocess sharing.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p ptr does not point to the
 * first byte of an allocation made through ROCr, or len is not the full length
 * of the allocation or handle is NULL.
 */
hsa_status_t HSA_API hsa_amd_ipc_memory_create(void* ptr, size_t len,
                                               hsa_amd_ipc_memory_t* handle);

/**
 * @brief Imports shared memory into the local process and makes it accessible
 * by the given agents.  If a shared memory handle is attached multiple times
 * in a process each attach may return a different address.  Each returned
 * address is refcounted and requires a matching number of calls to
 * hsa_amd_ipc_memory_detach to release the shared memory mapping.
 *
 * @param[in] handle Pointer to the identifier for the shared memory.
 *
 * @param[in] len Length of the shared memory to import.
 * Reserved.  Must be the full length of the shared allocation in this version.
 *
 * @param[in] num_agents Count of agents in @p mapping_agents.
 * May be zero if all agents are to be allowed access.
 *
 * @param[in] mapping_agents List of agents to access the shared memory.
 * Ignored if @p num_agents is zero.
 *
 * @param[out] mapped_ptr Recieves a process local pointer to the shared memory.
 *
 * @retval HSA_STATUS_SUCCESS if memory is successfully imported.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p handle is not valid, @p len is
 * incorrect, @p mapped_ptr is NULL, or some agent for which access was
 * requested can not access the shared memory.
 */
hsa_status_t HSA_API hsa_amd_ipc_memory_attach(
    const hsa_amd_ipc_memory_t* handle, size_t len,
    uint32_t num_agents,
    const hsa_agent_t* mapping_agents,
    void** mapped_ptr);

/**
 * @brief Decrements the reference count for the shared memory mapping and
 * releases access to shared memory imported with hsa_amd_ipc_memory_attach.
 *
 * @param[in] mapped_ptr Pointer to the first byte of a shared allocation
 * imported with hsa_amd_ipc_memory_attach.
 *
 * @retval HSA_STATUS_SUCCESS if @p mapped_ptr was imported with
 * hsa_amd_ipc_memory_attach.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p mapped_ptr was not imported
 * with hsa_amd_ipc_memory_attach.
 */
hsa_status_t HSA_API hsa_amd_ipc_memory_detach(void* mapped_ptr);

/**
 * @brief 256-bit process independent identifier for a ROCr IPC signal.
 */
typedef hsa_amd_ipc_memory_t hsa_amd_ipc_signal_t;

/**
 * @brief Obtains an interprocess sharing handle for a signal.  The handle is
 * valid while the signal it references remains valid in any process.  In
 * general applications should confirm that the signal has been attached (via
 * hsa_amd_ipc_signal_attach) in the remote process prior to destroying that
 * signal in the local process.
 * Repeated calls for the same signal may, but are not required to, return
 * unique handles.
 *
 * @param[in] signal Signal created with attribute HSA_AMD_SIGNAL_IPC.
 *
 * @param[out] handle Process independent identifier referencing the shared
 * signal.
 *
 * @retval HSA_STATUS_SUCCESS @p handle is ready to use for interprocess sharing.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p signal is not a valid signal
 * created with attribute HSA_AMD_SIGNAL_IPC or handle is NULL.
 */
hsa_status_t HSA_API hsa_amd_ipc_signal_create(hsa_signal_t signal, hsa_amd_ipc_signal_t* handle);

/**
 * @brief Imports an IPC capable signal into the local process.  If an IPC
 * signal handle is attached multiple times in a process each attach may return
 * a different signal handle.  Each returned signal handle is refcounted and
 * requires a matching number of calls to hsa_signal_destroy to release the
 * shared signal.
 *
 * @param[in] handle Pointer to the identifier for the shared signal.
 *
 * @param[out] signal Recieves a process local signal handle to the shared signal.
 *
 * @retval HSA_STATUS_SUCCESS if the signal is successfully imported.
 *
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED if HSA is not initialized
 *
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p handle is not valid.
 */
hsa_status_t HSA_API hsa_amd_ipc_signal_attach(const hsa_amd_ipc_signal_t* handle,
                                               hsa_signal_t* signal);

/**
 * @brief GPU system event type.
 */
typedef enum hsa_amd_event_type_s {
  /*
   AMD GPU memory fault.
   */
  HSA_AMD_GPU_MEMORY_FAULT_EVENT = 0,
  /*
   AMD GPU HW Exception.
   */
  HSA_AMD_GPU_HW_EXCEPTION_EVENT,
} hsa_amd_event_type_t;

/**
 * @brief Flags denoting the cause of a memory fault.
 */
typedef enum {
  // Page not present or supervisor privilege.
  HSA_AMD_MEMORY_FAULT_PAGE_NOT_PRESENT = 1 << 0,
  // Write access to a read-only page.
  HSA_AMD_MEMORY_FAULT_READ_ONLY = 1 << 1,
  // Execute access to a page marked NX.
  HSA_AMD_MEMORY_FAULT_NX = 1 << 2,
  // GPU attempted access to a host only page.
  HSA_AMD_MEMORY_FAULT_HOST_ONLY = 1 << 3,
  // DRAM ECC failure.
  HSA_AMD_MEMORY_FAULT_DRAMECC = 1 << 4,
  // Can't determine the exact fault address.
  HSA_AMD_MEMORY_FAULT_IMPRECISE = 1 << 5,
  // SRAM ECC failure (ie registers, no fault address).
  HSA_AMD_MEMORY_FAULT_SRAMECC = 1 << 6,
  // GPU reset following unspecified hang.
  HSA_AMD_MEMORY_FAULT_HANG = 1U << 31
} hsa_amd_memory_fault_reason_t;

/**
 * @brief AMD GPU memory fault event data.
 */
typedef struct hsa_amd_gpu_memory_fault_info_s {
  /*
  The agent where the memory fault occurred.
  */
  hsa_agent_t agent;
  /*
  Virtual address accessed.
  */
  uint64_t virtual_address;
  /*
  Bit field encoding the memory access failure reasons. There could be multiple bits set
  for one fault.  Bits are defined in hsa_amd_memory_fault_reason_t.
  */
  uint32_t fault_reason_mask;
} hsa_amd_gpu_memory_fault_info_t;

/**
 * @brief Flags denoting the type of a HW exception
 */
typedef enum {
  // Unused for now
  HSA_AMD_HW_EXCEPTION_RESET_TYPE_OTHER = 1 << 0,
} hsa_amd_hw_exception_reset_type_t;

/**
 * @brief Flags denoting the cause of a HW exception
 */
typedef enum {
  // GPU Hang
  HSA_AMD_HW_EXCEPTION_CAUSE_GPU_HANG = 1 << 0,
  // SRAM ECC
  HSA_AMD_HW_EXCEPTION_CAUSE_ECC = 1 << 1,
} hsa_amd_hw_exception_reset_cause_t;

/**
 * @brief AMD GPU HW Exception event data.
 */
typedef struct hsa_amd_gpu_hw_exception_info_s {
  /*
  The agent where the HW exception occurred.
  */
  hsa_agent_t agent;
  hsa_amd_hw_exception_reset_type_t reset_type;
  hsa_amd_hw_exception_reset_cause_t reset_cause;
} hsa_amd_gpu_hw_exception_info_t;

/**
 * @brief AMD GPU event data passed to event handler.
 */
typedef struct hsa_amd_event_s {
  /*
  The event type.
  */
  hsa_amd_event_type_t event_type;
  union {
    /*
    The memory fault info, only valid when @p event_type is HSA_AMD_GPU_MEMORY_FAULT_EVENT.
    */
    hsa_amd_gpu_memory_fault_info_t memory_fault;
    /*
    The memory fault info, only valid when @p event_type is HSA_AMD_GPU_HW_EXCEPTION_EVENT.
    */
    hsa_amd_gpu_hw_exception_info_t hw_exception;
  };
} hsa_amd_event_t;

typedef hsa_status_t (*hsa_amd_system_event_callback_t)(const hsa_amd_event_t* event, void* data);

/**
 * @brief Register AMD GPU event handler.
 *
 * @param[in] callback Callback to be invoked when an event is triggered.
 * The HSA runtime passes two arguments to the callback: @p event
 * is defined per event by the HSA runtime, and @p data is the user data.
 *
 * @param[in] data User data that is passed to @p callback. May be NULL.
 *
 * @retval HSA_STATUS_SUCCESS The handler has been registered successfully.
 *
 * @retval HSA_STATUS_ERROR An event handler has already been registered.
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT @p event is invalid.
 */
hsa_status_t HSA_API hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback,
                                                   void* data);

/**
 * @brief Per-queue dispatch and wavefront scheduling priority.
 */
typedef enum hsa_amd_queue_priority_s {
  /*
  Below normal/high priority compute and all graphics
  */
  HSA_AMD_QUEUE_PRIORITY_LOW = 0,
  /*
  Above low priority compute, below high priority compute and all graphics
  */
  HSA_AMD_QUEUE_PRIORITY_NORMAL = 1,
  /*
  Above low/normal priority compute and all graphics
  */
  HSA_AMD_QUEUE_PRIORITY_HIGH = 2,
} hsa_amd_queue_priority_t;

/**
 * @brief Modifies the dispatch and wavefront scheduling prioirty for a
 * given compute queue. The default is HSA_AMD_QUEUE_PRIORITY_NORMAL.
 *
 * @param[in] queue Compute queue to apply new priority to.
 *
 * @param[in] priority Priority to associate with queue.
 *
 * @retval HSA_STATUS_SUCCESS if priority was changed successfully.
 *
 * @retval HSA_STATUS_ERROR_INVALID_QUEUE if queue is not a valid
 * compute queue handle.
 *
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT if priority is not a valid
 * value from hsa_amd_queue_priority_t.
 */
hsa_status_t HSA_API hsa_amd_queue_set_priority(hsa_queue_t* queue,
                                                hsa_amd_queue_priority_t priority);

/**
 * @brief Deallocation notifier function type.
 */
typedef void (*hsa_amd_deallocation_callback_t)(void* ptr, void* user_data);

/**
 * @brief Registers a deallocation notifier monitoring for release of agent
 * accessible address @p ptr.  If successful, @p callback will be invoked when
 * @p ptr is removed from accessibility from all agents.
 *
 * Notification callbacks are automatically deregistered when they are invoked.
 *
 * Note: The current version supports notifications of address release
 * originating from ::hsa_amd_memory_pool_free.  Support for other address
 * release APIs will follow.
 *
 * @param[in] ptr Agent accessible address to monitor for deallocation.  Passed
 * to @p callback.
 *
 * @param[in] callback Notifier to be invoked when @p ptr is released from
 * agent accessibility.
 *
 * @param[in] user_data User provided value passed to @p callback.  May be NULL.
 *
 * @retval ::HSA_STATUS_SUCCESS The notifier registered successfully
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION @p ptr does not refer to a valid agent accessible
 * address.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT @p callback is NULL or @p ptr is NULL.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES if there is a failure in allocating
 * necessary resources
 */
hsa_status_t HSA_API hsa_amd_register_deallocation_callback(void* ptr,
                                                    hsa_amd_deallocation_callback_t callback,
                                                    void* user_data);

/**
 * @brief Removes a deallocation notifier previously registered with
 * ::hsa_amd_register_deallocation_callback.  Arguments must be identical to
 * those given in ::hsa_amd_register_deallocation_callback.
 *
 * @param[in] ptr Agent accessible address which was monitored for deallocation.
 *
 * @param[in] callback Notifier to be removed.
 *
 * @retval ::HSA_STATUS_SUCCESS The notifier has been removed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT The given notifier was not registered.
 */
hsa_status_t HSA_API hsa_amd_deregister_deallocation_callback(void* ptr,
                                                      hsa_amd_deallocation_callback_t callback);

typedef enum hsa_amd_svm_model_s {
  /**
   * Updates to memory with this attribute conform to HSA memory consistency
   * model.
   */
  HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED = 0,
  /**
   * Writes to memory with this attribute can be performed by a single agent
   * at a time.
   */
  HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED = 1,
  /**
   * Memory region queried contains subregions with both
   * HSA_AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED and
   * HSA_AMD_SVM_GLOBAL_FLAG_FINE_GRAINED attributes.
   *
   * This attribute can not be used in hsa_amd_svm_attributes_set.  It is a
   * possible return from hsa_amd_svm_attributes_get indicating that the query
   * region contains both coarse and fine grained memory.
   */
  HSA_AMD_SVM_GLOBAL_FLAG_INDETERMINATE = 2
} hsa_amd_svm_model_t;

typedef enum hsa_amd_svm_attribute_s {
  // Memory model attribute.
  // Type of this attribute is hsa_amd_svm_model_t.
  HSA_AMD_SVM_ATTRIB_GLOBAL_FLAG = 0,
  // Marks the range read only.  This allows multiple physical copies to be
  // placed local to each accessing device.
  // Type of this attribute is bool.
  HSA_AMD_SVM_ATTRIB_READ_ONLY = 1,
  // Automatic migrations should attempt to keep the memory within the xgmi hive
  // containing accessible agents.
  // Type of this attribute is bool.
  HSA_AMD_SVM_ATTRIB_HIVE_LOCAL = 2,
  // Page granularity to migrate at once.  Page granularity is specified as
  // log2(page_count).
  // Type of this attribute is uint64_t.
  HSA_AMD_SVM_ATTRIB_MIGRATION_GRANULARITY = 3,
  // Physical location to prefer when automatic migration occurs.
  // Set to the null agent handle (handle == 0) to indicate there
  // is no preferred location.
  // Type of this attribute is hsa_agent_t.
  HSA_AMD_SVM_ATTRIB_PREFERRED_LOCATION = 4,
  // This attribute can not be used in ::hsa_amd_svm_attributes_set (see
  // ::hsa_amd_svm_prefetch_async).
  // Queries the physical location of most recent prefetch command.
  // If the prefetch location has not been set or is not uniform across the
  // address range then returned hsa_agent_t::handle will be 0.
  // Querying this attribute will return the destination agent of the most
  // recent ::hsa_amd_svm_prefetch_async targeting the address range.  If
  // multiple async prefetches have been issued targeting the region and the
  // most recently issued prefetch has completed then the query will return
  // the location of the most recently completed prefetch.
  // Type of this attribute is hsa_agent_t.
  HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION = 5,
  // Optimizes with the anticipation that the majority of operations to the
  // range will be read operations.
  // Type of this attribute is bool.
  HSA_AMD_SVM_ATTRIB_READ_MOSTLY = 6,
  // Allows the execution on GPU.
  // Type of this attribute is bool.
  HSA_AMD_SVM_ATTRIB_GPU_EXEC = 7,
  // This attribute can not be used in ::hsa_amd_svm_attributes_get.
  // Enables an agent for access to the range.  Access may incur a page fault
  // and associated memory migration.  Either this or
  // HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE is required prior to SVM
  // access if HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT is false.
  // Type of this attribute is hsa_agent_t.
  HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE = 0x200,
  // This attribute can not be used in ::hsa_amd_svm_attributes_get.
  // Enables an agent for access to the range without page faults.  Access
  // will not incur a page fault and will not cause access based migration.
  // and associated memory migration.  Either this or
  // HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE is required prior to SVM access if
  // HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT is false.
  // Type of this attribute is hsa_agent_t.
  HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE = 0x201,
  // This attribute can not be used in ::hsa_amd_svm_attributes_get.
  // Denies an agent access to the memory range.  Access will cause a terminal
  // segfault.
  // Type of this attribute is hsa_agent_t.
  HSA_AMD_SVM_ATTRIB_AGENT_NO_ACCESS = 0x202,
  // This attribute can not be used in ::hsa_amd_svm_attributes_set.
  // Returns the access attribute associated with the agent.
  // The agent to query must be set in the attribute value field.
  // The attribute enum will be replaced with the agent's current access
  // attribute for the address range.
  // TODO: Clarify KFD return value for non-uniform access attribute.
  // Type of this attribute is hsa_agent_t.
  HSA_AMD_SVM_ATTRIB_ACCESS_QUERY = 0x203,
} hsa_amd_svm_attribute_t;

// List type for hsa_amd_svm_attributes_set/get. 
typedef struct hsa_amd_svm_attribute_pair_s {
  // hsa_amd_svm_attribute_t value.
  uint64_t attribute;
  // Attribute value.  Bit values should be interpreted according to the type
  // given in the associated attribute description.
  uint64_t value;
} hsa_amd_svm_attribute_pair_t;

/**
 * @brief Sets SVM memory attributes.
 *
 * If HSA_AMD_SYSTEM_INFO_SVM_ACCESSIBLE_BY_DEFAULT returns false then enabling
 * access to an Agent via this API (setting HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
 * or HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE) is required prior to SVM
 * memory access by that Agent.
 *
 * Attributes HSA_AMD_SVM_ATTRIB_ACCESS_QUERY and HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION
 * may not be used with this API.
 *
 * @param[in] ptr Will be aligned down to nearest page boundary.
 *
 * @param[in] size Will be aligned up to nearest page boundary.
 *
 * @param[in] attribute_list List of attributes to set for the address range.
 *
 * @param[in] attribute_count Length of @p attribute_list.
 */
hsa_status_t hsa_amd_svm_attributes_set(void* ptr, size_t size,
                                        hsa_amd_svm_attribute_pair_t* attribute_list,
                                        size_t attribute_count);

/**
 * @brief Gets SVM memory attributes.
 *
 * Attributes HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE,
 * HSA_AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE and
 * HSA_AMD_SVM_ATTRIB_PREFETCH_LOCATION may not be used with this API.
 *
 * Note that attribute HSA_AMD_SVM_ATTRIB_ACCESS_QUERY takes as input an
 * hsa_agent_t and returns the current access type through its attribute field.
 *
 * @param[in] ptr Will be aligned down to nearest page boundary.
 *
 * @param[in] size Will be aligned up to nearest page boundary.
 *
 * @param[in] attribute_list List of attributes to set for the address range.
 *
 * @param[in] attribute_count Length of @p attribute_list.
 */
hsa_status_t hsa_amd_svm_attributes_get(void* ptr, size_t size,
                                        hsa_amd_svm_attribute_pair_t* attribute_list,
                                        size_t attribute_count);

/**
 * @brief Asynchronously migrates memory to an agent.
 *
 * Schedules memory migration to @p agent when @p dep_signals have been observed equal to zero.
 * @p completion_signal will decrement when the migration is complete.
 *
 * @param[in] ptr Will be aligned down to nearest page boundary.
 *
 * @param[in] size Will be aligned up to nearest page boundary.
 *
 * @param[in] agent Agent to migrate to.
 *
 * @param[in] num_dep_signals Number of dependent signals. Can be 0.
 *
 * @param[in] dep_signals List of signals that must be waited on before the migration
 * operation starts. The migration will start after every signal has been observed with
 * the value 0. If @p num_dep_signals is 0, this argument is ignored.
 *
 * @param[in] completion_signal Signal used to indicate completion of the migration
 * operation. When the migration operation is finished, the value of the signal is
 * decremented. The runtime indicates that an error has occurred during the copy
 * operation by setting the value of the completion signal to a negative
 * number. If no completion signal is required this handle may be null.
 */
hsa_status_t hsa_amd_svm_prefetch_async(void* ptr, size_t size, hsa_agent_t agent,
                                        uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                        hsa_signal_t completion_signal);

/**
 * @brief Acquire Stream Performance Monitor on an agent
 *
 * Acquire exclusive use of SPM on @p preferred_agent.
 * See hsa_amd_spm_set_dest_buffer to provide a destination buffer to KFD to start recording and
 * retrieve this data.
 * @param[in] preferred_agent Agent on which to acquire SPM
 */
hsa_status_t hsa_amd_spm_acquire(hsa_agent_t preferred_agent);

/**
 * @brief Release Stream Performance Monitor on an agent
 *
 * Release exclusive use of SPM on @p preferred_agent. This will stop KFD writing SPM data.
 * If a destination buffer is set, then data in the destination buffer is available to user
 * when this function returns.
 *
 * @param[in] preferred_agent Agent on which to release SPM
 */
hsa_status_t hsa_amd_spm_release(hsa_agent_t preferred_agent);

/**
 * @brief  Set up the current destination user mode buffer for stream performance
 * counter data. KFD will start writing SPM data into the destination buffer. KFD will continue
 * to copy data into the current destination buffer until any of the following functions are called
 * - hsa_amd_spm_release
 * - hsa_amd_spm_set_dest_buffer with dest set to NULL
 * - hsa_amd_spm_set_dest_buffer with dest set to a new buffer
 *
 * if @p timeout is non-0, the call will wait for up to @p timeout ms for the previous
 * buffer to be filled. If previous buffer to be filled before timeout, the @p timeout
 * will be updated value with the time remaining. If the timeout is exceeded, the function
 * copies any partial data available into the previous user buffer and returns success.
 * User should not access destination data while KFD is copying data.
 * If the previous destination buffer was full, then @p is_data_loss flag is set.
 * @p dest is CPU accessible memory. It could be malloc'ed memory or host allocated memory
 *
 * @param[in] preferred_agent Agent on which to set the dest buffer
 *
 * @param[in] size_in_bytes size of the buffer
 *
 * @param[in/out] timeout timeout in milliseconds
 *
 * @param[out] size_copied number of bytes copied
 *
 * @param[in] dest destination address. Set to NULL to stop copy on previous buffer
 *
 * @param[out] is_data_loss true is data was lost
 */
hsa_status_t hsa_amd_spm_set_dest_buffer(hsa_agent_t preferred_agent, size_t size_in_bytes,
                                         uint32_t* timeout, uint32_t* size_copied, void* dest,
                                         bool* is_data_loss);
/**
 * @brief Obtains an OS specific, vendor neutral, handle to a memory allocation.
 *
 * Obtains an OS specific handle to GPU agent memory.  The memory must be part
 * of a single allocation from an hsa_amd_memory_pool_t exposed by a GPU Agent.
 * The handle may be used with other APIs (e.g. Vulkan) to obtain shared access
 * to the allocation.
 *
 * Shared access to the memory is not guaranteed to be fine grain coherent even
 * if the allocation exported is from a fine grain pool.  The shared memory
 * consistency model will be no stronger than the model exported from, consult
 * the importing API to determine the final consistency model.
 *
 * The allocation's memory remains valid as long as the handle and any mapping
 * of the handle remains valid.  When the handle and all mappings are closed
 * the backing memory will be released for reuse.
 *
 * @param[in] ptr Pointer to the allocation being exported.
 *
 * @param[in] size Size in bytes to export following @p ptr.  The entire range
 * being exported must be contained within a single allocation.
 *
 * @param[out] dmabuf Pointer to a dma-buf file descriptor holding a reference to the
 * allocation.  Contents will not be altered in the event of failure.
 *
 * @param[out] offset Offset in bytes into the memory referenced by the dma-buf
 * object at which @p ptr resides.  Contents will not be altered in the event
 * of failure.
 *
 * @retval ::HSA_STATUS_SUCCESS Export completed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT One or more arguments is NULL.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION The address range described by
 * @p ptr and @p size are not contained within a single allocation.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The allocation described by @p ptr
 * and @p size was allocated on a device which can not export memory.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES The return file descriptor,
 * @p dmabuf, could not be created.
 */
hsa_status_t hsa_amd_portable_export_dmabuf(const void* ptr, size_t size, int* dmabuf,
                                            uint64_t* offset);

/**
 * @brief Closes an OS specific, vendor neutral, handle to a memory allocation.
 *
 * Closes an OS specific handle to GPU agent memory.
 *
 * Applications should close a handle after imports are complete.  The handle
 * is not required to remain open for the lifetime of imported mappings.  The
 * referenced allocation will remain valid until all handles and mappings
 * are closed.
 *
 * @param[in] dmabuf Handle to be closed.
 *
 * @retval ::HSA_STATUS_SUCCESS Handle closed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_RESOURCE_FREE A generic error was encountered
 * when closing the handle.  The handle may have been closed already or an
 * async IO error may have occured.
 */
hsa_status_t hsa_amd_portable_close_dmabuf(int dmabuf);

/**
 * @brief Allocate a reserved address range
 *
 * Reserve a virtual address range. The size must be a multiple of the system page size.
 * If it is not possible to allocate the address specified by @p address, then @p va will be
 * a different address range.
 * Address range should be released by calling hsa_amd_vmem_address_free.
 *
 * @param[out] va virtual address allocated
 * @param[in] size of address range requested
 * @param[in] address requested
 * @param[in] flags currently unsupported
 *
 * @retval ::HSA_STATUS_SUCCESS Address range allocated successfully
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Insufficient resources to allocate an address
 * range of this size.
 *
 * Note that this API will be deprecated in a future release and replaced by
 * hsa_amd_vmem_address_reserve_align
 */
hsa_status_t hsa_amd_vmem_address_reserve(void** va, size_t size, uint64_t address,
                                          uint64_t flags);

/**
 * @brief Allocate a reserved address range
 *
 * Reserve a virtual address range. The size must be a multiple of the system page size.
 * If it is not possible to allocate the address specified by @p address, then @p va will be
 * a different address range.
 * Address range should be released by calling hsa_amd_vmem_address_free.
 *
 * @param[out] va virtual address allocated
 * @param[in] size of address range requested
 * @param[in] address requested
 * @param[in] alignment requested. 0 for default. Must be >= page-size and a power of 2
 * @param[in] flags currently unsupported
 *
 * @retval ::HSA_STATUS_SUCCESS Address range allocated successfully
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Insufficient resources to allocate an address
 * range of this size.
 */
hsa_status_t hsa_amd_vmem_address_reserve_align(void** va, size_t size, uint64_t address,
                                          uint64_t alignment, uint64_t flags);

/**
 * @brief Free a reserved address range
 *
 * Free a previously allocated address range. The size must match the size of a previously
 * allocated address range.
 *
 * @param[out] va virtual address to be freed
 * @param[in] size of address range
 *
 * @retval ::HSA_STATUS_SUCCESS Address range released successfully
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION Invalid va specified
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid size specified
 * @retval ::HSA_STATUS_ERROR_RESOURCE_FREE Address range is still in use
 * @retval ::HSA_STATUS_ERROR Internal unexpected error
 */
hsa_status_t hsa_amd_vmem_address_free(void* va, size_t size);

/**
 * @brief Struct containing an opaque handle to a memory allocation handle
 */
typedef struct hsa_amd_vmem_alloc_handle_s {
  /**
   * Opaque handle. Two handles reference the same object of the enclosing type
   * if and only if they are equal.
   */
  uint64_t handle;
} hsa_amd_vmem_alloc_handle_t;

typedef enum {
  MEMORY_TYPE_NONE,
  MEMORY_TYPE_PINNED,
} hsa_amd_memory_type_t;

/**
 * @brief Create a virtual memory handle
 *
 * Create a virtual memory handle within this pool
 * @p size must be a aligned to allocation granule size for this memory pool, see
 * HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE
 * To minimize internal memory fragmentation, align the size to the recommended allocation granule
 * size, see HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE
 *
 * @param[in] pool memory to use
 * @param[in] size of the memory allocation
 * @param[in] type of memory
 * @param[in] flags - currently unsupported
 * @param[out] memory_handle - handle for the allocation
 *
 * @retval ::HSA_STATUS_SUCCESS memory allocated successfully
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid arguments
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION This memory pool does not support allocations
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Insufficient resources to allocate this memory
 */
hsa_status_t hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t pool, size_t size,
                                        hsa_amd_memory_type_t type, uint64_t flags,
                                        hsa_amd_vmem_alloc_handle_t* memory_handle);

/**
 * @brief Release a virtual memory handle
 *
 * @param[in] memory handle that was previously allocated
 *
 * @retval ::HSA_STATUS_SUCCESS Address range allocated successfully
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION Invalid memory handle
 */
hsa_status_t hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t memory_handle);

/**
 * @brief Map a virtual memory handle
 *
 * Map a virtual memory handle to a reserved address range. The virtual address requested must be
 * within a previously reserved address range. @p va and (@p va + size) must be must be within
 * (va + size) of the previous allocated address range.
 * @p size must be equal to size of the @p memory_handle
 * hsa_amd_vmem_set_access needs to be called to make the memory accessible to specific agents
 *
 * @param[in] va virtual address range where memory will be mapped
 * @param[in] size of memory mapping
 * @param[in] in_offset offset into memory. Currently unsupported
 * @param[in] memory_handle virtual memory handle to be mapped
 * @param[in] flags. Currently unsupported
 *
 * @retval ::HSA_STATUS_SUCCESS Memory mapped successfully
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT va, size or memory_handle are invalid
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Insufficient resources
 *
 * @retval ::HSA_STATUS_ERROR Unexpected internal error
 */
hsa_status_t hsa_amd_vmem_map(void* va, size_t size, size_t in_offset,
                              hsa_amd_vmem_alloc_handle_t memory_handle, uint64_t flags);

/**
 * @brief Unmap a virtual memory handle
 *
 * Unmap previously mapped virtual address range
 *
 * @param[in] va virtual address range where memory will be mapped
 * @param[in] size of memory mapping
 *
 * @retval ::HSA_STATUS_SUCCESS Memory backing unmapped successfully
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION memory_handle is invalid
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT size is invalid
 *
 * @retval ::HSA_STATUS_ERROR Unexpected internal error
 */
hsa_status_t hsa_amd_vmem_unmap(void* va, size_t size);

typedef struct hsa_amd_memory_access_desc_s {
  hsa_access_permission_t permissions;
  hsa_agent_t agent_handle;
} hsa_amd_memory_access_desc_t;

/**
 * @brief Make a memory mapping accessible
 *
 * Make previously mapped virtual address accessible to specific agents. @p size must be equal to
 * size of previously mapped virtual memory handle.
 * Calling hsa_amd_vmem_set_access multiple times on the same @p va will overwrite previous
 * permissions for all agents
 *
 * @param[in] va previously mapped virtual address
 * @param[in] size of memory mapping
 * @param[in] desc list of access permissions for each agent
 * @param[in] desc_cnt number of elements in desc
 *
 * @retval ::HSA_STATUS_SUCCESS
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT va, size or memory_handle are invalid
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION memory_handle is invalid
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Insufficient resources
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT Invalid agent in desc
 *
 * @retval ::HSA_STATUS_ERROR Unexpected internal error
 */
hsa_status_t hsa_amd_vmem_set_access(void* va, size_t size,
                                     const hsa_amd_memory_access_desc_t* desc,
                                     size_t desc_cnt);

/**
 * @brief Get current access permissions for memory mapping
 *
 * Get access permissions for memory mapping for specific agent.
 *
 * @param[in] va previously mapped virtual address
 * @param[in] perms current permissions
 * @param[in] agent_handle agent
 *
 * @retval ::HSA_STATUS_SUCCESS
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT Invalid agent
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION va is not mapped or permissions never set for this
 * agent
 *
 * @retval ::HSA_STATUS_ERROR Unexpected internal error
 */
hsa_status_t hsa_amd_vmem_get_access(void* va, hsa_access_permission_t* perms,
                                     hsa_agent_t agent_handle);

/**
 * @brief Get an exportable shareable handle
 *
 * Get an exportable shareable handle for a memory_handle. This shareabl handle can then be used to
 * re-create a virtual memory handle using hsa_amd_vmem_import_shareable_handle. The shareable
 * handle can be transferred using mechanisms that support posix file descriptors Once all shareable
 * handles are closed, the memory_handle is released.
 *
 * @param[out] dmabuf_fd shareable handle
 * @param[in] handle previously allocated virtual memory handle
 * @param[in] flags Currently unsupported
 *
 * @retval ::HSA_STATUS_SUCCESS
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION Invalid memory handle
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Out of resources
 *
 * @retval ::HSA_STATUS_ERROR Unexpected internal error
 */
hsa_status_t hsa_amd_vmem_export_shareable_handle(int* dmabuf_fd,
                                                  hsa_amd_vmem_alloc_handle_t handle,
                                                  uint64_t flags);
/**
 * @brief Import a shareable handle
 *
 * Import a shareable handle for a memory handle. Importing a shareable handle that has been closed
 * and released results in undefined behavior.
 *
 * @param[in] dmabuf_fd shareable handle exported with hsa_amd_vmem_export_shareable_handle
 * @param[out] handle virtual memory handle
 *
 * @retval ::HSA_STATUS_SUCCESS
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION Invalid memory handle
 *
 * @retval ::HSA_STATUS_ERROR_OUT_OF_RESOURCES Out of resources
 *
 * @retval ::HSA_STATUS_ERROR Unexpected internal error
 */
hsa_status_t hsa_amd_vmem_import_shareable_handle(int dmabuf_fd,
                                                  hsa_amd_vmem_alloc_handle_t* handle);

/**
 * @brief Returns memory handle for mapped memory
 *
 * Return a memory handle for previously mapped memory. The handle will be the same value of handle
 * used to map the memory. The returned handle must be released with corresponding number of calls
 * to hsa_amd_vmem_handle_release.
 *
 * @param[out] memory_handle memory handle for this mapped address
 * @param[in] mapped address
 *
 * @retval ::HSA_STATUS_SUCCESS
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION Invalid address
 */
hsa_status_t hsa_amd_vmem_retain_alloc_handle(hsa_amd_vmem_alloc_handle_t* memory_handle,
                                              void* addr);

/**
 * @brief Returns the current allocation properties of a handle
 *
 * Returns the allocation properties of an existing handle
 *
 * @param[in] memory_handle memory handle to be queried
 * @param[out] pool memory pool that owns this handle
 * @param[out] memory type

 * @retval ::HSA_STATUS_SUCCESS
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ALLOCATION Invalid memory_handle
 */
hsa_status_t hsa_amd_vmem_get_alloc_properties_from_handle(
    hsa_amd_vmem_alloc_handle_t memory_handle, hsa_amd_memory_pool_t* pool,
    hsa_amd_memory_type_t* type);

/**
 * @brief Set the asynchronous scratch limit threshold on all the queues for this agent.
 * Dispatches that are enqueued on HW queues on this agent that are smaller than threshold will not
 * result in a scratch use-once method.
 *
 * Increasing this threshold will only increase the internal limit and not cause immediate allocation
 * of additional scratch memory. Decreasing this threshold will result in a release in scratch memory
 * on queues where the current amount of allocated scratch exceeds the new limit.
 *
 * This API is only supported on devices that support asynchronous scratch reclaim.
 *
 * @param[in] agent A valid agent.
 *
 * @param[in] threshold Threshold size in bytes
 *
 * @retval ::HSA_STATUS_SUCCESS The function has been executed successfully.
 *
 * @retval ::HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime has not been
 * initialized.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_AGENT The agent is invalid.
 *
 * @retval ::HSA_STATUS_ERROR_INVALID_ARGUMENT This agent does not support asynchronous scratch
 * reclaim
 */
hsa_status_t HSA_API hsa_amd_agent_set_async_scratch_limit(hsa_agent_t agent, size_t threshold);

typedef enum {
  /*
   * Returns the agent that owns the underlying HW queue.
   * The type of this attribute is hsa_agent_t.
   */
  HSA_AMD_QUEUE_INFO_AGENT,
  /*
   * Returns the doorbell ID of the completion signal of the queue
   * The type of this attribute is uint64_t.
   */
  HSA_AMD_QUEUE_INFO_DOORBELL_ID,
} hsa_queue_info_attribute_t;

hsa_status_t hsa_amd_queue_get_info(hsa_queue_t* queue, hsa_queue_info_attribute_t attribute,
                                    void* value);

#ifdef __cplusplus
}  // end extern "C" block
#endif

#endif  // header guard
