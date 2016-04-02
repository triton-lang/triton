/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include "isaac/driver/common.h"
#include "isaac/exception/driver.h"

namespace isaac
{
namespace driver
{

void check(nvrtcResult err)
{
  using namespace isaac::exception::nvrtc;

  switch(err)
  {
    case NVRTC_SUCCESS:                         break;
    case NVRTC_ERROR_OUT_OF_MEMORY:             throw out_of_memory();
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:  throw program_creation_failure();
    case NVRTC_ERROR_INVALID_INPUT:             throw invalid_input();
    case NVRTC_ERROR_INVALID_PROGRAM:           throw invalid_program();
    case NVRTC_ERROR_INVALID_OPTION:            throw invalid_option();
    case NVRTC_ERROR_COMPILATION:               throw compilation();
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE: throw builtin_operation_failure();
    default: throw unknown_error();
  }
}

void check(CUresult err)
{
  using namespace isaac::exception::cuda;
  switch(err)
  {
    case CUDA_SUCCESS                              : break;
    case CUDA_ERROR_INVALID_VALUE                  : throw invalid_value();
    case CUDA_ERROR_OUT_OF_MEMORY                  : throw out_of_memory();
    case CUDA_ERROR_NOT_INITIALIZED                : throw not_initialized();
    case CUDA_ERROR_DEINITIALIZED                  : throw deinitialized();
    case CUDA_ERROR_PROFILER_DISABLED              : throw profiler_disabled();
    case CUDA_ERROR_PROFILER_NOT_INITIALIZED       : throw profiler_not_initialized();
    case CUDA_ERROR_PROFILER_ALREADY_STARTED       : throw profiler_already_started();
    case CUDA_ERROR_PROFILER_ALREADY_STOPPED       : throw profiler_already_stopped();
    case CUDA_ERROR_NO_DEVICE                      : throw no_device();
    case CUDA_ERROR_INVALID_DEVICE                 : throw invalid_device();
    case CUDA_ERROR_INVALID_IMAGE                  : throw invalid_image();
    case CUDA_ERROR_INVALID_CONTEXT                : throw invalid_context();
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT        : throw context_already_current();
    case CUDA_ERROR_MAP_FAILED                     : throw map_failed();
    case CUDA_ERROR_UNMAP_FAILED                   : throw unmap_failed();
    case CUDA_ERROR_ARRAY_IS_MAPPED                : throw array_is_mapped();
    case CUDA_ERROR_ALREADY_MAPPED                 : throw already_mapped();
    case CUDA_ERROR_NO_BINARY_FOR_GPU              : throw no_binary_for_gpu();
    case CUDA_ERROR_ALREADY_ACQUIRED               : throw already_acquired();
    case CUDA_ERROR_NOT_MAPPED                     : throw not_mapped();
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY            : throw not_mapped_as_array();
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER          : throw not_mapped_as_pointer();
    case CUDA_ERROR_ECC_UNCORRECTABLE              : throw ecc_uncorrectable();
    case CUDA_ERROR_UNSUPPORTED_LIMIT              : throw unsupported_limit();
    case CUDA_ERROR_CONTEXT_ALREADY_IN_USE         : throw context_already_in_use();
    case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED        : throw peer_access_unsupported();
    case CUDA_ERROR_INVALID_PTX                    : throw invalid_ptx();
    case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT       : throw invalid_graphics_context();
    case CUDA_ERROR_INVALID_SOURCE                 : throw invalid_source();
    case CUDA_ERROR_FILE_NOT_FOUND                 : throw file_not_found();
    case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND : throw shared_object_symbol_not_found();
    case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      : throw shared_object_init_failed();
    case CUDA_ERROR_OPERATING_SYSTEM               : throw operating_system();
    case CUDA_ERROR_INVALID_HANDLE                 : throw invalid_handle();
    case CUDA_ERROR_NOT_FOUND                      : throw not_found();
    case CUDA_ERROR_NOT_READY                      : throw not_ready();
    case CUDA_ERROR_ILLEGAL_ADDRESS                : throw illegal_address();
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        : throw launch_out_of_resources();
    case CUDA_ERROR_LAUNCH_TIMEOUT                 : throw launch_timeout();
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  : throw launch_incompatible_texturing();
    case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    : throw peer_access_already_enabled();
    case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        : throw peer_access_not_enabled();
    case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         : throw primary_context_active();
    case CUDA_ERROR_CONTEXT_IS_DESTROYED           : throw context_is_destroyed();
    case CUDA_ERROR_ASSERT                         : throw assert_error();
    case CUDA_ERROR_TOO_MANY_PEERS                 : throw too_many_peers();
    case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED : throw host_memory_already_registered();
    case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     : throw host_memory_not_registered();
    case CUDA_ERROR_HARDWARE_STACK_ERROR           : throw hardware_stack_error();
    case CUDA_ERROR_ILLEGAL_INSTRUCTION            : throw illegal_instruction();
    case CUDA_ERROR_MISALIGNED_ADDRESS             : throw misaligned_address();
    case CUDA_ERROR_INVALID_ADDRESS_SPACE          : throw invalid_address_space();
    case CUDA_ERROR_INVALID_PC                     : throw invalid_pc();
    case CUDA_ERROR_LAUNCH_FAILED                  : throw launch_failed();
    case CUDA_ERROR_NOT_PERMITTED                  : throw not_permitted();
    case CUDA_ERROR_NOT_SUPPORTED                  : throw not_supported();
    case CUDA_ERROR_UNKNOWN                        : throw unknown();
    default                                        : throw unknown();
  }
}

void check_destruction(CUresult result)
{
    if(result!=CUDA_ERROR_DEINITIALIZED)
        check(result);
}


void check(cl_int err)
{
    using namespace isaac::exception::ocl;
    switch(err)
    {
        case CL_SUCCESS:                        break;
        case CL_DEVICE_NOT_FOUND:               throw device_not_found();
        case CL_DEVICE_NOT_AVAILABLE:           throw device_not_available();
        case CL_COMPILER_NOT_AVAILABLE:         throw compiler_not_available();
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:  throw mem_object_allocation_failure();
        case CL_OUT_OF_RESOURCES:               throw out_of_resources();
        case CL_OUT_OF_HOST_MEMORY:             throw out_of_host_memory();
        case CL_PROFILING_INFO_NOT_AVAILABLE:   throw profiling_info_not_available();
        case CL_MEM_COPY_OVERLAP:               throw mem_copy_overlap();
        case CL_IMAGE_FORMAT_MISMATCH:          throw image_format_mismatch();
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:     throw image_format_not_supported();
        case CL_BUILD_PROGRAM_FAILURE:          throw build_program_failure();
        case CL_MAP_FAILURE:                    throw map_failure();

        case CL_INVALID_VALUE:                  throw invalid_value();
        case CL_INVALID_DEVICE_TYPE:            throw invalid_device_type();
        case CL_INVALID_PLATFORM:               throw invalid_platform();
        case CL_INVALID_DEVICE:                 throw invalid_device();
        case CL_INVALID_CONTEXT:                throw invalid_context();
        case CL_INVALID_QUEUE_PROPERTIES:       throw invalid_queue_properties();
        case CL_INVALID_COMMAND_QUEUE:          throw invalid_command_queue();
        case CL_INVALID_HOST_PTR:               throw invalid_host_ptr();
        case CL_INVALID_MEM_OBJECT:             throw invalid_mem_object();
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: throw invalid_image_format_descriptor();
        case CL_INVALID_IMAGE_SIZE:             throw invalid_image_size();
        case CL_INVALID_SAMPLER:                throw invalid_sampler();
        case CL_INVALID_BINARY:                 throw invalid_binary();
        case CL_INVALID_BUILD_OPTIONS:          throw invalid_build_options();
        case CL_INVALID_PROGRAM:                throw invalid_program();
        case CL_INVALID_PROGRAM_EXECUTABLE:     throw invalid_program_executable();
        case CL_INVALID_KERNEL_NAME:            throw invalid_kernel_name();
        case CL_INVALID_KERNEL_DEFINITION:      throw invalid_kernel_definition();
        case CL_INVALID_KERNEL:                 throw invalid_kernel();
        case CL_INVALID_ARG_INDEX:              throw invalid_arg_index();
        case CL_INVALID_ARG_VALUE:              throw invalid_arg_value();
        case CL_INVALID_ARG_SIZE:               throw invalid_arg_size();
        case CL_INVALID_KERNEL_ARGS:            throw invalid_kernel_args();
        case CL_INVALID_WORK_DIMENSION:         throw invalid_work_dimension();
        case CL_INVALID_WORK_GROUP_SIZE:        throw invalid_work_group_size();
        case CL_INVALID_WORK_ITEM_SIZE:         throw invalid_work_item_size();
        case CL_INVALID_GLOBAL_OFFSET:          throw invalid_global_offset();
        case CL_INVALID_EVENT_WAIT_LIST:        throw invalid_event_wait_list();
        case CL_INVALID_EVENT:                  throw invalid_event();
        case CL_INVALID_OPERATION:              throw invalid_operation();
        case CL_INVALID_GL_OBJECT:              throw invalid_gl_object();
        case CL_INVALID_BUFFER_SIZE:            throw invalid_buffer_size();
        case CL_INVALID_MIP_LEVEL:              throw invalid_mip_level();
        case CL_INVALID_GLOBAL_WORK_SIZE:       throw invalid_global_work_size();
    #ifdef CL_INVALID_PROPERTY
        case CL_INVALID_PROPERTY:               throw invalid_property();
    #endif
        default: throw;
    }
}

}
}

