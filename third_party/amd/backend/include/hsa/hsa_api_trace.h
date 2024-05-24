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

#ifndef HSA_RUNTIME_INC_HSA_API_TRACE_H
#define HSA_RUNTIME_INC_HSA_API_TRACE_H

#include "hsa.h"
#ifdef AMD_INTERNAL_BUILD
#include "hsa_ext_image.h"
#include "hsa_ext_amd.h"
#include "hsa_ext_finalize.h"
#include "hsa_amd_tool.h"
#else
#include "inc/hsa_ext_image.h"
#include "inc/hsa_ext_amd.h"
#include "inc/hsa_ext_finalize.h"
#include "inc/hsa_amd_tool.h"
#endif

#include <string.h>
#include <assert.h>
#include <stddef.h>

// Major Ids of the Api tables exported by Hsa Core Runtime
#define HSA_API_TABLE_MAJOR_VERSION               0x03
#define HSA_CORE_API_TABLE_MAJOR_VERSION          0x02
#define HSA_AMD_EXT_API_TABLE_MAJOR_VERSION       0x02
#define HSA_FINALIZER_API_TABLE_MAJOR_VERSION     0x02
#define HSA_IMAGE_API_TABLE_MAJOR_VERSION         0x02
#define HSA_AQLPROFILE_API_TABLE_MAJOR_VERSION    0x01
#define HSA_TOOLS_API_TABLE_MAJOR_VERSION         0x01

// Step Ids of the Api tables exported by Hsa Core Runtime
#define HSA_API_TABLE_STEP_VERSION                0x00
#define HSA_CORE_API_TABLE_STEP_VERSION           0x00
#define HSA_AMD_EXT_API_TABLE_STEP_VERSION        0x01
#define HSA_FINALIZER_API_TABLE_STEP_VERSION      0x00
#define HSA_IMAGE_API_TABLE_STEP_VERSION          0x00
#define HSA_AQLPROFILE_API_TABLE_STEP_VERSION     0x00
#define HSA_TOOLS_API_TABLE_STEP_VERSION          0x00

// Min function used to copy Api Tables
static inline uint32_t Min(const uint32_t a, const uint32_t b) {
  return (a > b) ? b : a;
}

// Declarations of APIs intended for use only by tools.

// An AQL packet that can be put in an intercept queue to cause a callback to
// be invoked when the packet is about to be submitted to the underlying
// hardware queue. These packets are not copied to the underlying hardware
// queue. These packets should come immediately before the regular AQL packet
// they relate to. This implies that packet rewriters should always keep these
// packets adjacent to the regular AQL packet that follows them.
const uint32_t AMD_AQL_FORMAT_INTERCEPT_MARKER = 0xFE;

struct amd_aql_intercept_marker_s;

// When an intercept queue is processing rewritten packets to put them on the
// underlying hardware queue, if it encounters a
// AMD_AQL_FORMAT_INTERCEPT_MARKER vendor AQL packet it will call the following
// handler. packet points to the packet, queue is the underlying hardware
// queue, and packet_id is the packet id of the next packet to be put on the
// underlying hardware queue. The intercept queue does not put these packets
// onto the underlying hardware queue.
typedef void (*amd_intercept_marker_handler)(const struct amd_aql_intercept_marker_s* packet,
                                             hsa_queue_t* queue, uint64_t packet_id);
// An AQL vendor packet used by the intercept queue to mark the following
// packet. The callback will be invoked to allow a tool to know where in the
// underlying hardware queue the following packet will be placed. user_data can
// be used to hold any data useful to the tool.
typedef struct amd_aql_intercept_marker_s {
  uint16_t header; // Must have a packet type of HSA_PACKET_TYPE_VENDOR_SPECIFIC.
  uint8_t format; // Must be AMD_AQL_FORMAT_INTERCEPT_MARKER.
  uint8_t reserved[5]; // Must be 0.
#ifdef HSA_LARGE_MODEL
  amd_intercept_marker_handler callback;
#elif defined HSA_LITTLE_ENDIAN
  amd_intercept_marker_handler callback;
  uint32_t reserved1; // Must be 0.
#else
  uint32_t reserved1; // Must be 0.
  amd_intercept_marker_handler callback;
#endif
  uint64_t user_data[6];
} amd_aql_intercept_marker_t;

typedef void (*hsa_amd_queue_intercept_packet_writer)(const void* pkts, uint64_t pkt_count);
typedef void (*hsa_amd_queue_intercept_handler)(const void* pkts, uint64_t pkt_count,
                                                uint64_t user_pkt_index, void* data,
                                                hsa_amd_queue_intercept_packet_writer writer);
hsa_status_t hsa_amd_queue_intercept_register(hsa_queue_t* queue,
                                              hsa_amd_queue_intercept_handler callback,
                                              void* user_data);
hsa_status_t hsa_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue);

typedef void (*hsa_amd_runtime_queue_notifier)(const hsa_queue_t* queue, hsa_agent_t agent,
                                               void* data);
hsa_status_t hsa_amd_runtime_queue_create_register(hsa_amd_runtime_queue_notifier callback,
                                                   void* user_data);

// Structure of Version used to identify an instance of Api table
// Must be the first member (offsetof == 0) of all API tables.
// This is the root of the table passing ABI.
struct ApiTableVersion {
  uint32_t major_id;
  uint32_t minor_id;
  uint32_t step_id;
  uint32_t reserved;
};

struct ToolsApiTable {
  ApiTableVersion version;

  hsa_amd_tool_event hsa_amd_tool_scratch_event_alloc_start_fn;
  hsa_amd_tool_event hsa_amd_tool_scratch_event_alloc_end_fn;
  hsa_amd_tool_event hsa_amd_tool_scratch_event_free_start_fn;
  hsa_amd_tool_event hsa_amd_tool_scratch_event_free_end_fn;
  hsa_amd_tool_event hsa_amd_tool_scratch_event_async_reclaim_start_fn;
  hsa_amd_tool_event hsa_amd_tool_scratch_event_async_reclaim_end_fn;
};

// Table to export HSA Finalizer Extension Apis
struct FinalizerExtTable {
  ApiTableVersion version;
	decltype(hsa_ext_program_create)* hsa_ext_program_create_fn;
	decltype(hsa_ext_program_destroy)* hsa_ext_program_destroy_fn;
	decltype(hsa_ext_program_add_module)* hsa_ext_program_add_module_fn;
	decltype(hsa_ext_program_iterate_modules)* hsa_ext_program_iterate_modules_fn;
	decltype(hsa_ext_program_get_info)* hsa_ext_program_get_info_fn;
	decltype(hsa_ext_program_finalize)* hsa_ext_program_finalize_fn;
};

// Table to export HSA Image Extension Apis
struct ImageExtTable {
  ApiTableVersion version;
	decltype(hsa_ext_image_get_capability)* hsa_ext_image_get_capability_fn;
	decltype(hsa_ext_image_data_get_info)* hsa_ext_image_data_get_info_fn;
	decltype(hsa_ext_image_create)* hsa_ext_image_create_fn;
	decltype(hsa_ext_image_import)* hsa_ext_image_import_fn;
	decltype(hsa_ext_image_export)* hsa_ext_image_export_fn;
	decltype(hsa_ext_image_copy)* hsa_ext_image_copy_fn;
	decltype(hsa_ext_image_clear)* hsa_ext_image_clear_fn;
	decltype(hsa_ext_image_destroy)* hsa_ext_image_destroy_fn;
	decltype(hsa_ext_sampler_create)* hsa_ext_sampler_create_fn;
	decltype(hsa_ext_sampler_destroy)* hsa_ext_sampler_destroy_fn;
  decltype(hsa_ext_image_get_capability_with_layout)* hsa_ext_image_get_capability_with_layout_fn;
  decltype(hsa_ext_image_data_get_info_with_layout)* hsa_ext_image_data_get_info_with_layout_fn;
  decltype(hsa_ext_image_create_with_layout)* hsa_ext_image_create_with_layout_fn;
};

// Table to export AMD Extension Apis
struct AmdExtTable {
  ApiTableVersion version;
	decltype(hsa_amd_coherency_get_type)* hsa_amd_coherency_get_type_fn;
	decltype(hsa_amd_coherency_set_type)* hsa_amd_coherency_set_type_fn;
  decltype(hsa_amd_profiling_set_profiler_enabled)* hsa_amd_profiling_set_profiler_enabled_fn;
  decltype(hsa_amd_profiling_async_copy_enable) *hsa_amd_profiling_async_copy_enable_fn;
  decltype(hsa_amd_profiling_get_dispatch_time)* hsa_amd_profiling_get_dispatch_time_fn;
  decltype(hsa_amd_profiling_get_async_copy_time) *hsa_amd_profiling_get_async_copy_time_fn;
  decltype(hsa_amd_profiling_convert_tick_to_system_domain)* hsa_amd_profiling_convert_tick_to_system_domain_fn;
  decltype(hsa_amd_signal_async_handler)* hsa_amd_signal_async_handler_fn;
  decltype(hsa_amd_async_function)* hsa_amd_async_function_fn;
  decltype(hsa_amd_signal_wait_any)* hsa_amd_signal_wait_any_fn;
  decltype(hsa_amd_queue_cu_set_mask)* hsa_amd_queue_cu_set_mask_fn;
  decltype(hsa_amd_memory_pool_get_info)* hsa_amd_memory_pool_get_info_fn;
  decltype(hsa_amd_agent_iterate_memory_pools)* hsa_amd_agent_iterate_memory_pools_fn;
  decltype(hsa_amd_memory_pool_allocate)* hsa_amd_memory_pool_allocate_fn;
  decltype(hsa_amd_memory_pool_free)* hsa_amd_memory_pool_free_fn;
  decltype(hsa_amd_memory_async_copy)* hsa_amd_memory_async_copy_fn;
  decltype(hsa_amd_memory_async_copy_on_engine)* hsa_amd_memory_async_copy_on_engine_fn;
  decltype(hsa_amd_memory_copy_engine_status)* hsa_amd_memory_copy_engine_status_fn;
  decltype(hsa_amd_agent_memory_pool_get_info)* hsa_amd_agent_memory_pool_get_info_fn;
  decltype(hsa_amd_agents_allow_access)* hsa_amd_agents_allow_access_fn;
  decltype(hsa_amd_memory_pool_can_migrate)* hsa_amd_memory_pool_can_migrate_fn;
  decltype(hsa_amd_memory_migrate)* hsa_amd_memory_migrate_fn;
  decltype(hsa_amd_memory_lock)* hsa_amd_memory_lock_fn;
  decltype(hsa_amd_memory_unlock)* hsa_amd_memory_unlock_fn;
  decltype(hsa_amd_memory_fill)* hsa_amd_memory_fill_fn;
  decltype(hsa_amd_interop_map_buffer)* hsa_amd_interop_map_buffer_fn;
  decltype(hsa_amd_interop_unmap_buffer)* hsa_amd_interop_unmap_buffer_fn;
  decltype(hsa_amd_image_create)* hsa_amd_image_create_fn;
  decltype(hsa_amd_pointer_info)* hsa_amd_pointer_info_fn;
  decltype(hsa_amd_pointer_info_set_userdata)* hsa_amd_pointer_info_set_userdata_fn;
  decltype(hsa_amd_ipc_memory_create)* hsa_amd_ipc_memory_create_fn;
  decltype(hsa_amd_ipc_memory_attach)* hsa_amd_ipc_memory_attach_fn;
  decltype(hsa_amd_ipc_memory_detach)* hsa_amd_ipc_memory_detach_fn;
  decltype(hsa_amd_signal_create)* hsa_amd_signal_create_fn;
  decltype(hsa_amd_ipc_signal_create)* hsa_amd_ipc_signal_create_fn;
  decltype(hsa_amd_ipc_signal_attach)* hsa_amd_ipc_signal_attach_fn;
  decltype(hsa_amd_register_system_event_handler)* hsa_amd_register_system_event_handler_fn;
  decltype(hsa_amd_queue_intercept_create)* hsa_amd_queue_intercept_create_fn;
  decltype(hsa_amd_queue_intercept_register)* hsa_amd_queue_intercept_register_fn;
  decltype(hsa_amd_queue_set_priority)* hsa_amd_queue_set_priority_fn;
  decltype(hsa_amd_memory_async_copy_rect)* hsa_amd_memory_async_copy_rect_fn;
  decltype(hsa_amd_runtime_queue_create_register)* hsa_amd_runtime_queue_create_register_fn;
  decltype(hsa_amd_memory_lock_to_pool)* hsa_amd_memory_lock_to_pool_fn;
  decltype(hsa_amd_register_deallocation_callback)* hsa_amd_register_deallocation_callback_fn;
  decltype(hsa_amd_deregister_deallocation_callback)* hsa_amd_deregister_deallocation_callback_fn;
  decltype(hsa_amd_signal_value_pointer)* hsa_amd_signal_value_pointer_fn;
  decltype(hsa_amd_svm_attributes_set)* hsa_amd_svm_attributes_set_fn;
  decltype(hsa_amd_svm_attributes_get)* hsa_amd_svm_attributes_get_fn;
  decltype(hsa_amd_svm_prefetch_async)* hsa_amd_svm_prefetch_async_fn;
  decltype(hsa_amd_spm_acquire)* hsa_amd_spm_acquire_fn;
  decltype(hsa_amd_spm_release)* hsa_amd_spm_release_fn;
  decltype(hsa_amd_spm_set_dest_buffer)* hsa_amd_spm_set_dest_buffer_fn;
  decltype(hsa_amd_queue_cu_get_mask)* hsa_amd_queue_cu_get_mask_fn;
  decltype(hsa_amd_portable_export_dmabuf)* hsa_amd_portable_export_dmabuf_fn;
  decltype(hsa_amd_portable_close_dmabuf)* hsa_amd_portable_close_dmabuf_fn;
  decltype(hsa_amd_vmem_address_reserve)* hsa_amd_vmem_address_reserve_fn;
  decltype(hsa_amd_vmem_address_free)* hsa_amd_vmem_address_free_fn;
  decltype(hsa_amd_vmem_handle_create)* hsa_amd_vmem_handle_create_fn;
  decltype(hsa_amd_vmem_handle_release)* hsa_amd_vmem_handle_release_fn;
  decltype(hsa_amd_vmem_map)* hsa_amd_vmem_map_fn;
  decltype(hsa_amd_vmem_unmap)* hsa_amd_vmem_unmap_fn;
  decltype(hsa_amd_vmem_set_access)* hsa_amd_vmem_set_access_fn;
  decltype(hsa_amd_vmem_get_access)* hsa_amd_vmem_get_access_fn;
  decltype(hsa_amd_vmem_export_shareable_handle)* hsa_amd_vmem_export_shareable_handle_fn;
  decltype(hsa_amd_vmem_import_shareable_handle)* hsa_amd_vmem_import_shareable_handle_fn;
  decltype(hsa_amd_vmem_retain_alloc_handle)* hsa_amd_vmem_retain_alloc_handle_fn;
  decltype(hsa_amd_vmem_get_alloc_properties_from_handle)*
      hsa_amd_vmem_get_alloc_properties_from_handle_fn;
  decltype(hsa_amd_agent_set_async_scratch_limit)* hsa_amd_agent_set_async_scratch_limit_fn;
};

// Table to export HSA Core Runtime Apis
struct CoreApiTable {
  ApiTableVersion version;
  decltype(hsa_init)* hsa_init_fn;
  decltype(hsa_shut_down)* hsa_shut_down_fn;
  decltype(hsa_system_get_info)* hsa_system_get_info_fn;
  decltype(hsa_system_extension_supported)* hsa_system_extension_supported_fn;
  decltype(hsa_system_get_extension_table)* hsa_system_get_extension_table_fn;
  decltype(hsa_iterate_agents)* hsa_iterate_agents_fn;
  decltype(hsa_agent_get_info)* hsa_agent_get_info_fn;
  decltype(hsa_queue_create)* hsa_queue_create_fn;
  decltype(hsa_soft_queue_create)* hsa_soft_queue_create_fn;
  decltype(hsa_queue_destroy)* hsa_queue_destroy_fn;
  decltype(hsa_queue_inactivate)* hsa_queue_inactivate_fn;
  decltype(hsa_queue_load_read_index_scacquire)* hsa_queue_load_read_index_scacquire_fn;
  decltype(hsa_queue_load_read_index_relaxed)* hsa_queue_load_read_index_relaxed_fn;
  decltype(hsa_queue_load_write_index_scacquire)* hsa_queue_load_write_index_scacquire_fn;
  decltype(hsa_queue_load_write_index_relaxed)* hsa_queue_load_write_index_relaxed_fn;
  decltype(hsa_queue_store_write_index_relaxed)* hsa_queue_store_write_index_relaxed_fn;
  decltype(hsa_queue_store_write_index_screlease)* hsa_queue_store_write_index_screlease_fn;
  decltype(hsa_queue_cas_write_index_scacq_screl)* hsa_queue_cas_write_index_scacq_screl_fn;
  decltype(hsa_queue_cas_write_index_scacquire)* hsa_queue_cas_write_index_scacquire_fn;
  decltype(hsa_queue_cas_write_index_relaxed)* hsa_queue_cas_write_index_relaxed_fn;
  decltype(hsa_queue_cas_write_index_screlease)* hsa_queue_cas_write_index_screlease_fn;
  decltype(hsa_queue_add_write_index_scacq_screl)* hsa_queue_add_write_index_scacq_screl_fn;
  decltype(hsa_queue_add_write_index_scacquire)* hsa_queue_add_write_index_scacquire_fn;
  decltype(hsa_queue_add_write_index_relaxed)* hsa_queue_add_write_index_relaxed_fn;
  decltype(hsa_queue_add_write_index_screlease)* hsa_queue_add_write_index_screlease_fn;
  decltype(hsa_queue_store_read_index_relaxed)* hsa_queue_store_read_index_relaxed_fn;
  decltype(hsa_queue_store_read_index_screlease)* hsa_queue_store_read_index_screlease_fn;
  decltype(hsa_agent_iterate_regions)* hsa_agent_iterate_regions_fn;
  decltype(hsa_region_get_info)* hsa_region_get_info_fn;
  decltype(hsa_agent_get_exception_policies)* hsa_agent_get_exception_policies_fn;
  decltype(hsa_agent_extension_supported)* hsa_agent_extension_supported_fn;
  decltype(hsa_memory_register)* hsa_memory_register_fn;
  decltype(hsa_memory_deregister)* hsa_memory_deregister_fn;
  decltype(hsa_memory_allocate)* hsa_memory_allocate_fn;
  decltype(hsa_memory_free)* hsa_memory_free_fn;
  decltype(hsa_memory_copy)* hsa_memory_copy_fn;
  decltype(hsa_memory_assign_agent)* hsa_memory_assign_agent_fn;
  decltype(hsa_signal_create)* hsa_signal_create_fn;
  decltype(hsa_signal_destroy)* hsa_signal_destroy_fn;
  decltype(hsa_signal_load_relaxed)* hsa_signal_load_relaxed_fn;
  decltype(hsa_signal_load_scacquire)* hsa_signal_load_scacquire_fn;
  decltype(hsa_signal_store_relaxed)* hsa_signal_store_relaxed_fn;
  decltype(hsa_signal_store_screlease)* hsa_signal_store_screlease_fn;
  decltype(hsa_signal_wait_relaxed)* hsa_signal_wait_relaxed_fn;
  decltype(hsa_signal_wait_scacquire)* hsa_signal_wait_scacquire_fn;
  decltype(hsa_signal_and_relaxed)* hsa_signal_and_relaxed_fn;
  decltype(hsa_signal_and_scacquire)* hsa_signal_and_scacquire_fn;
  decltype(hsa_signal_and_screlease)* hsa_signal_and_screlease_fn;
  decltype(hsa_signal_and_scacq_screl)* hsa_signal_and_scacq_screl_fn;
  decltype(hsa_signal_or_relaxed)* hsa_signal_or_relaxed_fn;
  decltype(hsa_signal_or_scacquire)* hsa_signal_or_scacquire_fn;
  decltype(hsa_signal_or_screlease)* hsa_signal_or_screlease_fn;
  decltype(hsa_signal_or_scacq_screl)* hsa_signal_or_scacq_screl_fn;
  decltype(hsa_signal_xor_relaxed)* hsa_signal_xor_relaxed_fn;
  decltype(hsa_signal_xor_scacquire)* hsa_signal_xor_scacquire_fn;
  decltype(hsa_signal_xor_screlease)* hsa_signal_xor_screlease_fn;
  decltype(hsa_signal_xor_scacq_screl)* hsa_signal_xor_scacq_screl_fn;
  decltype(hsa_signal_exchange_relaxed)* hsa_signal_exchange_relaxed_fn;
  decltype(hsa_signal_exchange_scacquire)* hsa_signal_exchange_scacquire_fn;
  decltype(hsa_signal_exchange_screlease)* hsa_signal_exchange_screlease_fn;
  decltype(hsa_signal_exchange_scacq_screl)* hsa_signal_exchange_scacq_screl_fn;
  decltype(hsa_signal_add_relaxed)* hsa_signal_add_relaxed_fn;
  decltype(hsa_signal_add_scacquire)* hsa_signal_add_scacquire_fn;
  decltype(hsa_signal_add_screlease)* hsa_signal_add_screlease_fn;
  decltype(hsa_signal_add_scacq_screl)* hsa_signal_add_scacq_screl_fn;
  decltype(hsa_signal_subtract_relaxed)* hsa_signal_subtract_relaxed_fn;
  decltype(hsa_signal_subtract_scacquire)* hsa_signal_subtract_scacquire_fn;
  decltype(hsa_signal_subtract_screlease)* hsa_signal_subtract_screlease_fn;
  decltype(hsa_signal_subtract_scacq_screl)* hsa_signal_subtract_scacq_screl_fn;
  decltype(hsa_signal_cas_relaxed)* hsa_signal_cas_relaxed_fn;
  decltype(hsa_signal_cas_scacquire)* hsa_signal_cas_scacquire_fn;
  decltype(hsa_signal_cas_screlease)* hsa_signal_cas_screlease_fn;
  decltype(hsa_signal_cas_scacq_screl)* hsa_signal_cas_scacq_screl_fn;

  //===--- Instruction Set Architecture -----------------------------------===//

  decltype(hsa_isa_from_name)* hsa_isa_from_name_fn;
  // Deprecated since v1.1.
  decltype(hsa_isa_get_info)* hsa_isa_get_info_fn;
  // Deprecated since v1.1.
  decltype(hsa_isa_compatible)* hsa_isa_compatible_fn;

  //===--- Code Objects (deprecated) --------------------------------------===//

  // Deprecated since v1.1.
  decltype(hsa_code_object_serialize)* hsa_code_object_serialize_fn;
  // Deprecated since v1.1.
  decltype(hsa_code_object_deserialize)* hsa_code_object_deserialize_fn;
  // Deprecated since v1.1.
  decltype(hsa_code_object_destroy)* hsa_code_object_destroy_fn;
  // Deprecated since v1.1.
  decltype(hsa_code_object_get_info)* hsa_code_object_get_info_fn;
  // Deprecated since v1.1.
  decltype(hsa_code_object_get_symbol)* hsa_code_object_get_symbol_fn;
  // Deprecated since v1.1.
  decltype(hsa_code_symbol_get_info)* hsa_code_symbol_get_info_fn;
  // Deprecated since v1.1.
  decltype(hsa_code_object_iterate_symbols)* hsa_code_object_iterate_symbols_fn;

  //===--- Executable -----------------------------------------------------===//

  // Deprecated since v1.1.
  decltype(hsa_executable_create)* hsa_executable_create_fn;
  decltype(hsa_executable_destroy)* hsa_executable_destroy_fn;
  // Deprecated since v1.1.
  decltype(hsa_executable_load_code_object)* hsa_executable_load_code_object_fn;
  decltype(hsa_executable_freeze)* hsa_executable_freeze_fn;
  decltype(hsa_executable_get_info)* hsa_executable_get_info_fn;
  decltype(hsa_executable_global_variable_define)*
      hsa_executable_global_variable_define_fn;
  decltype(hsa_executable_agent_global_variable_define)*
      hsa_executable_agent_global_variable_define_fn;
  decltype(hsa_executable_readonly_variable_define)*
      hsa_executable_readonly_variable_define_fn;
  decltype(hsa_executable_validate)* hsa_executable_validate_fn;
  // Deprecated since v1.1.
  decltype(hsa_executable_get_symbol)* hsa_executable_get_symbol_fn;
  decltype(hsa_executable_symbol_get_info)* hsa_executable_symbol_get_info_fn;
  // Deprecated since v1.1.
  decltype(hsa_executable_iterate_symbols)* hsa_executable_iterate_symbols_fn;

  //===--- Runtime Notifications ------------------------------------------===//

  decltype(hsa_status_string)* hsa_status_string_fn;

  // Start HSA v1.1 additions
  decltype(hsa_extension_get_name)* hsa_extension_get_name_fn;
  decltype(hsa_system_major_extension_supported)* hsa_system_major_extension_supported_fn;
  decltype(hsa_system_get_major_extension_table)* hsa_system_get_major_extension_table_fn;
  decltype(hsa_agent_major_extension_supported)* hsa_agent_major_extension_supported_fn;
  decltype(hsa_cache_get_info)* hsa_cache_get_info_fn;
  decltype(hsa_agent_iterate_caches)* hsa_agent_iterate_caches_fn;
  decltype(hsa_signal_silent_store_relaxed)* hsa_signal_silent_store_relaxed_fn;
  decltype(hsa_signal_silent_store_screlease)* hsa_signal_silent_store_screlease_fn;
  decltype(hsa_signal_group_create)* hsa_signal_group_create_fn;
  decltype(hsa_signal_group_destroy)* hsa_signal_group_destroy_fn;
  decltype(hsa_signal_group_wait_any_scacquire)* hsa_signal_group_wait_any_scacquire_fn;
  decltype(hsa_signal_group_wait_any_relaxed)* hsa_signal_group_wait_any_relaxed_fn;

  //===--- Instruction Set Architecture - HSA v1.1 additions --------------===//

  decltype(hsa_agent_iterate_isas)* hsa_agent_iterate_isas_fn;
  decltype(hsa_isa_get_info_alt)* hsa_isa_get_info_alt_fn;
  decltype(hsa_isa_get_exception_policies)* hsa_isa_get_exception_policies_fn;
  decltype(hsa_isa_get_round_method)* hsa_isa_get_round_method_fn;
  decltype(hsa_wavefront_get_info)* hsa_wavefront_get_info_fn;
  decltype(hsa_isa_iterate_wavefronts)* hsa_isa_iterate_wavefronts_fn;

  //===--- Code Objects (deprecated) - HSA v1.1 additions -----------------===//

  // Deprecated since v1.1.
  decltype(hsa_code_object_get_symbol_from_name)*
      hsa_code_object_get_symbol_from_name_fn;

  //===--- Executable - HSA v1.1 additions --------------------------------===//

  decltype(hsa_code_object_reader_create_from_file)*
      hsa_code_object_reader_create_from_file_fn;
  decltype(hsa_code_object_reader_create_from_memory)*
      hsa_code_object_reader_create_from_memory_fn;
  decltype(hsa_code_object_reader_destroy)* hsa_code_object_reader_destroy_fn;
  decltype(hsa_executable_create_alt)* hsa_executable_create_alt_fn;
  decltype(hsa_executable_load_program_code_object)*
      hsa_executable_load_program_code_object_fn;
  decltype(hsa_executable_load_agent_code_object)*
      hsa_executable_load_agent_code_object_fn;
  decltype(hsa_executable_validate_alt)* hsa_executable_validate_alt_fn;
  decltype(hsa_executable_get_symbol_by_name)*
      hsa_executable_get_symbol_by_name_fn;
  decltype(hsa_executable_iterate_agent_symbols)*
      hsa_executable_iterate_agent_symbols_fn;
  decltype(hsa_executable_iterate_program_symbols)*
      hsa_executable_iterate_program_symbols_fn;
};

// Table to export HSA Apis from Core Runtime, Amd Extensions
// Finalizer and Images
struct HsaApiTable {

  // Version of Hsa Api Table
  ApiTableVersion version;

  // Table of function pointers to HSA Core Runtime
	CoreApiTable* core_;

  // Table of function pointers to AMD extensions
	AmdExtTable* amd_ext_;

  // Table of function pointers to HSA Finalizer Extension
	FinalizerExtTable* finalizer_ext_;

  // Table of function pointers to HSA Image Extension
	ImageExtTable* image_ext_;

  // Table of function pointers for tools to use
  ToolsApiTable* tools_;
};

// Structure containing instances of different api tables
struct HsaApiTableContainer {
  HsaApiTable root;
	CoreApiTable core;
	AmdExtTable amd_ext;
	FinalizerExtTable finalizer_ext;
	ImageExtTable image_ext;
	ToolsApiTable tools;

  // Default initialization of a container instance
  HsaApiTableContainer() {
    root.version.major_id = HSA_API_TABLE_MAJOR_VERSION;
    root.version.minor_id = sizeof(HsaApiTable);
    root.version.step_id = HSA_API_TABLE_STEP_VERSION;

    core.version.major_id = HSA_CORE_API_TABLE_MAJOR_VERSION;
    core.version.minor_id = sizeof(CoreApiTable);
    core.version.step_id = HSA_CORE_API_TABLE_STEP_VERSION;
    root.core_ = &core;

    amd_ext.version.major_id = HSA_AMD_EXT_API_TABLE_MAJOR_VERSION;
    amd_ext.version.minor_id = sizeof(AmdExtTable);
    amd_ext.version.step_id = HSA_AMD_EXT_API_TABLE_STEP_VERSION;
    root.amd_ext_ = &amd_ext;

    finalizer_ext.version.major_id = HSA_FINALIZER_API_TABLE_MAJOR_VERSION;
    finalizer_ext.version.minor_id = sizeof(FinalizerExtTable);
    finalizer_ext.version.step_id = HSA_FINALIZER_API_TABLE_STEP_VERSION;
    root.finalizer_ext_ = &finalizer_ext;

    image_ext.version.major_id = HSA_IMAGE_API_TABLE_MAJOR_VERSION;
    image_ext.version.minor_id = sizeof(ImageExtTable);
    image_ext.version.step_id = HSA_IMAGE_API_TABLE_STEP_VERSION;
    root.image_ext_ = &image_ext;

    tools.version.major_id = HSA_TOOLS_API_TABLE_MAJOR_VERSION;
    tools.version.minor_id = sizeof(ToolsApiTable);
    tools.version.step_id = HSA_TOOLS_API_TABLE_STEP_VERSION;
    root.tools_ = &tools;
  }
};

// Api to copy function pointers of a table
static
void inline copyApi(void* src, void* dest, size_t size) {
  assert(size >= sizeof(ApiTableVersion));
  memcpy((char*)src + sizeof(ApiTableVersion),
         (char*)dest + sizeof(ApiTableVersion),
         (size - sizeof(ApiTableVersion)));
}

// Copy Api child tables if valid.
static void inline copyElement(ApiTableVersion* dest, ApiTableVersion* src) {
  if (src->major_id && (dest->major_id == src->major_id)) {
    dest->step_id = src->step_id;
    dest->minor_id = Min(dest->minor_id, src->minor_id);
    copyApi(dest, src, dest->minor_id);
  } else {
    dest->major_id = 0;
    dest->minor_id = 0;
    dest->step_id = 0;
  }
}

// Copy constructor for all Api tables. The function assumes the
// user has initialized an instance of tables container correctly
// for the Major, Minor and Stepping Ids of Root and Child Api tables.
// The function will overwrite the value of Minor Id by taking the
// minimum of source and destination parameters. It will also overwrite
// the stepping Id with value from source parameter.
static void inline copyTables(const HsaApiTable* src, HsaApiTable* dest) {
  // Verify Major Id of source and destination tables match
  if (dest->version.major_id != src->version.major_id) {
    dest->version.major_id = 0;
    dest->version.minor_id = 0;
    dest->version.step_id = 0;
    return;
  }

  // Initialize the stepping id and minor id of root table. For the
  // minor id which encodes struct size, take the minimum of source
  // and destination parameters
  dest->version.step_id = src->version.step_id;
  dest->version.minor_id = Min(dest->version.minor_id, src->version.minor_id);

  // Copy child tables if present
  if ((offsetof(HsaApiTable, core_) < dest->version.minor_id))
    copyElement(&dest->core_->version, &src->core_->version);
  if ((offsetof(HsaApiTable, amd_ext_) < dest->version.minor_id))
    copyElement(&dest->amd_ext_->version, &src->amd_ext_->version);
  if ((offsetof(HsaApiTable, finalizer_ext_) < dest->version.minor_id))
    copyElement(&dest->finalizer_ext_->version, &src->finalizer_ext_->version);
  if ((offsetof(HsaApiTable, image_ext_) < dest->version.minor_id))
    copyElement(&dest->image_ext_->version, &src->image_ext_->version);
  if ((offsetof(HsaApiTable, tools_) < dest->version.minor_id))
    copyElement(&dest->tools_->version, &src->tools_->version);
}
#endif
