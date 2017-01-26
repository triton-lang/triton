/* Copyright 2015-2017 Philippe Tillet
* 
* Permission is hereby granted, free of charge, to any person obtaining 
* a copy of this software and associated documentation files 
* (the "Software"), to deal in the Software without restriction, 
* including without limitation the rights to use, copy, modify, merge, 
* publish, distribute, sublicense, and/or sell copies of the Software, 
* and to permit persons to whom the Software is furnished to do so, 
* subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be 
* included in all copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef ISAAC_EXCEPTION_DRIVER_H
#define ISAAC_EXCEPTION_DRIVER_H
#include <exception>

#include "isaac/driver/dispatch.h"
#include "isaac/defines.h"

DISABLE_MSVC_WARNING_C4275

namespace isaac
{

namespace exception
{

  class ISAACAPI unknown_architecture: public std::exception{
    public:
      unknown_architecture(std::string const & msg): msg_("Unrecognized architecture: " + msg){}
      const char * what() const throw(){ return msg_.c_str(); }
    private:
      std::string msg_;
  };

namespace nvrtc
{

#define ISAAC_CREATE_NVRTC_EXCEPTION(name, msg) class ISAACAPI name: public std::exception { public: const char * what() const throw(){ return "NVRTC: Error- " msg; } }

  ISAAC_CREATE_NVRTC_EXCEPTION(out_of_memory              ,"out of memory");
  ISAAC_CREATE_NVRTC_EXCEPTION(program_creation_failure   ,"program creation failure");
  ISAAC_CREATE_NVRTC_EXCEPTION(invalid_input              ,"invalid input");
  ISAAC_CREATE_NVRTC_EXCEPTION(invalid_program            ,"invalid program");
  ISAAC_CREATE_NVRTC_EXCEPTION(invalid_option             ,"invalid option");
  ISAAC_CREATE_NVRTC_EXCEPTION(compilation                ,"compilation");
  ISAAC_CREATE_NVRTC_EXCEPTION(builtin_operation_failure  ,"builtin operation failure");
  ISAAC_CREATE_NVRTC_EXCEPTION(unknown_error              ,"unknown error");

  #undef ISAAC_CREATE_NVRTC_EXCEPTION
}


namespace cuda
{
    class base: public std::exception{};

#define ISAAC_CREATE_CUDA_EXCEPTION(name, msg) class ISAACAPI name: public base { public:const char * what() const throw(){ return "CUDA: Error- " msg; } }


    ISAAC_CREATE_CUDA_EXCEPTION(invalid_value                   ,"invalid value");
    ISAAC_CREATE_CUDA_EXCEPTION(out_of_memory                   ,"out of memory");
    ISAAC_CREATE_CUDA_EXCEPTION(not_initialized                 ,"not initialized");
    ISAAC_CREATE_CUDA_EXCEPTION(deinitialized                   ,"deinitialized");
    ISAAC_CREATE_CUDA_EXCEPTION(profiler_disabled               ,"profiler disabled");
    ISAAC_CREATE_CUDA_EXCEPTION(profiler_not_initialized        ,"profiler not initialized");
    ISAAC_CREATE_CUDA_EXCEPTION(profiler_already_started        ,"profiler already started");
    ISAAC_CREATE_CUDA_EXCEPTION(profiler_already_stopped        ,"profiler already stopped");
    ISAAC_CREATE_CUDA_EXCEPTION(no_device                       ,"no device");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_device                  ,"invalid device");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_image                   ,"invalid image");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_context                 ,"invalid context");
    ISAAC_CREATE_CUDA_EXCEPTION(context_already_current         ,"context already current");
    ISAAC_CREATE_CUDA_EXCEPTION(map_failed                      ,"map failed");
    ISAAC_CREATE_CUDA_EXCEPTION(unmap_failed                    ,"unmap failed");
    ISAAC_CREATE_CUDA_EXCEPTION(array_is_mapped                 ,"array is mapped");
    ISAAC_CREATE_CUDA_EXCEPTION(already_mapped                  ,"already mapped");
    ISAAC_CREATE_CUDA_EXCEPTION(no_binary_for_gpu               ,"no binary for gpu");
    ISAAC_CREATE_CUDA_EXCEPTION(already_acquired                ,"already acquired");
    ISAAC_CREATE_CUDA_EXCEPTION(not_mapped                      ,"not mapped");
    ISAAC_CREATE_CUDA_EXCEPTION(not_mapped_as_array             ,"not mapped as array");
    ISAAC_CREATE_CUDA_EXCEPTION(not_mapped_as_pointer           ,"not mapped as pointer");
    ISAAC_CREATE_CUDA_EXCEPTION(ecc_uncorrectable               ,"ecc uncorrectable");
    ISAAC_CREATE_CUDA_EXCEPTION(unsupported_limit               ,"unsupported limit");
    ISAAC_CREATE_CUDA_EXCEPTION(context_already_in_use          ,"context already in use");
    ISAAC_CREATE_CUDA_EXCEPTION(peer_access_unsupported         ,"peer access unsupported");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_ptx                     ,"invalid ptx");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_graphics_context        ,"invalid graphics context");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_source                  ,"invalid source");
    ISAAC_CREATE_CUDA_EXCEPTION(file_not_found                  ,"file not found");
    ISAAC_CREATE_CUDA_EXCEPTION(shared_object_symbol_not_found  ,"shared object symbol not found");
    ISAAC_CREATE_CUDA_EXCEPTION(shared_object_init_failed       ,"shared object init failed");
    ISAAC_CREATE_CUDA_EXCEPTION(operating_system                ,"operating system");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_handle                  ,"invalid handle");
    ISAAC_CREATE_CUDA_EXCEPTION(not_found                       ,"not found");
    ISAAC_CREATE_CUDA_EXCEPTION(not_ready                       ,"not ready");
    ISAAC_CREATE_CUDA_EXCEPTION(illegal_address                 ,"illegal address");
    ISAAC_CREATE_CUDA_EXCEPTION(launch_out_of_resources         ,"launch out of resources");
    ISAAC_CREATE_CUDA_EXCEPTION(launch_timeout                  ,"launch timeout");
    ISAAC_CREATE_CUDA_EXCEPTION(launch_incompatible_texturing   ,"launch incompatible texturing");
    ISAAC_CREATE_CUDA_EXCEPTION(peer_access_already_enabled     ,"peer access already enabled");
    ISAAC_CREATE_CUDA_EXCEPTION(peer_access_not_enabled         ,"peer access not enabled");
    ISAAC_CREATE_CUDA_EXCEPTION(primary_context_active          ,"primary context active");
    ISAAC_CREATE_CUDA_EXCEPTION(context_is_destroyed            ,"context is destroyed");
    ISAAC_CREATE_CUDA_EXCEPTION(assert_error                    ,"assert");
    ISAAC_CREATE_CUDA_EXCEPTION(too_many_peers                  ,"too many peers");
    ISAAC_CREATE_CUDA_EXCEPTION(host_memory_already_registered  ,"host memory already registered");
    ISAAC_CREATE_CUDA_EXCEPTION(host_memory_not_registered      ,"hot memory not registered");
    ISAAC_CREATE_CUDA_EXCEPTION(hardware_stack_error            ,"hardware stack error");
    ISAAC_CREATE_CUDA_EXCEPTION(illegal_instruction             ,"illegal instruction");
    ISAAC_CREATE_CUDA_EXCEPTION(misaligned_address              ,"misaligned address");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_address_space           ,"invalid address space");
    ISAAC_CREATE_CUDA_EXCEPTION(invalid_pc                      ,"invalid pc");
    ISAAC_CREATE_CUDA_EXCEPTION(launch_failed                   ,"launch failed");
    ISAAC_CREATE_CUDA_EXCEPTION(not_permitted                   ,"not permitted");
    ISAAC_CREATE_CUDA_EXCEPTION(not_supported                   ,"not supported");
    ISAAC_CREATE_CUDA_EXCEPTION(unknown                         ,"unknown");

    #undef ISAAC_CREATE_CUDA_EXCEPTION
}

namespace cublas
{

#define ISAAC_CREATE_CUBLAS_EXCEPTION(name, msg) class ISAACAPI name: public std::exception { public: const char * what() const throw(){ return "CUBLAS: Error- " msg; } }

  ISAAC_CREATE_CUBLAS_EXCEPTION(not_initialized              ,"not initialized");
  ISAAC_CREATE_CUBLAS_EXCEPTION(alloc_failed                 ,"alloc failed");
  ISAAC_CREATE_CUBLAS_EXCEPTION(invalid_value                ,"invalid value");
  ISAAC_CREATE_CUBLAS_EXCEPTION(arch_mismatch                ,"arch mismatch");
  ISAAC_CREATE_CUBLAS_EXCEPTION(mapping_error                ,"mapping error");
  ISAAC_CREATE_CUBLAS_EXCEPTION(execution_failed             ,"execution failed");
  ISAAC_CREATE_CUBLAS_EXCEPTION(internal_error               ,"internal error");
  ISAAC_CREATE_CUBLAS_EXCEPTION(not_supported                ,"not supported");
  ISAAC_CREATE_CUBLAS_EXCEPTION(license_error                ,"license error");
  ISAAC_CREATE_CUBLAS_EXCEPTION(unknown                      ,"unknown");

  #undef ISAAC_CREATE_CUBLAS_EXCEPTION
}

namespace ocl
{

    class ISAACAPI base: public std::exception{};

#define ISAAC_CREATE_CL_EXCEPTION(name, msg) class ISAACAPI name: public base { public: const char * what() const throw(){ return "OpenCL: Error- " msg; } }


   ISAAC_CREATE_CL_EXCEPTION(device_not_found,                  "device not found");
   ISAAC_CREATE_CL_EXCEPTION(device_not_available,              "device not available");
   ISAAC_CREATE_CL_EXCEPTION(compiler_not_available,            "compiler not available");
   ISAAC_CREATE_CL_EXCEPTION(mem_object_allocation_failure,     "object allocation failure");
   ISAAC_CREATE_CL_EXCEPTION(out_of_resources,                  "launch out of resources");
   ISAAC_CREATE_CL_EXCEPTION(out_of_host_memory,                "out of host memory");
   ISAAC_CREATE_CL_EXCEPTION(profiling_info_not_available,      "profiling info not available");
   ISAAC_CREATE_CL_EXCEPTION(mem_copy_overlap,                  "mem copy overlap");
   ISAAC_CREATE_CL_EXCEPTION(image_format_mismatch,             "image format mismatch");
   ISAAC_CREATE_CL_EXCEPTION(image_format_not_supported,        "image format not supported");
   ISAAC_CREATE_CL_EXCEPTION(build_program_failure,             "build program failure");
   ISAAC_CREATE_CL_EXCEPTION(map_failure,                       "map failure");
   ISAAC_CREATE_CL_EXCEPTION(invalid_value,                     "invalid value");
   ISAAC_CREATE_CL_EXCEPTION(invalid_device_type,               "invalid device type");
   ISAAC_CREATE_CL_EXCEPTION(invalid_platform,                  "invalid platform");
   ISAAC_CREATE_CL_EXCEPTION(invalid_device,                    "invalid device");
   ISAAC_CREATE_CL_EXCEPTION(invalid_context,                   "invalid context");
   ISAAC_CREATE_CL_EXCEPTION(invalid_queue_properties,          "invalid queue properties");
   ISAAC_CREATE_CL_EXCEPTION(invalid_command_queue,             "invalid command queue");
   ISAAC_CREATE_CL_EXCEPTION(invalid_host_ptr,                  "invalid host pointer");
   ISAAC_CREATE_CL_EXCEPTION(invalid_mem_object,                "invalid mem object");
   ISAAC_CREATE_CL_EXCEPTION(invalid_image_format_descriptor,   "invalid image format descriptor");
   ISAAC_CREATE_CL_EXCEPTION(invalid_image_size,                "invalid image size");
   ISAAC_CREATE_CL_EXCEPTION(invalid_sampler,                   "invalid sampler");
   ISAAC_CREATE_CL_EXCEPTION(invalid_binary,                    "invalid binary");
   ISAAC_CREATE_CL_EXCEPTION(invalid_build_options,             "invalid build options");
   ISAAC_CREATE_CL_EXCEPTION(invalid_program,                   "invalid program");
   ISAAC_CREATE_CL_EXCEPTION(invalid_program_executable,        "invalid program executable");
   ISAAC_CREATE_CL_EXCEPTION(invalid_kernel_name,               "invalid kernel name");
   ISAAC_CREATE_CL_EXCEPTION(invalid_kernel_definition,         "invalid kernel definition");
   ISAAC_CREATE_CL_EXCEPTION(invalid_kernel,                    "invalid kernel");
   ISAAC_CREATE_CL_EXCEPTION(invalid_arg_index,                 "invalid arg index");
   ISAAC_CREATE_CL_EXCEPTION(invalid_arg_value,                 "invalid arg value");
   ISAAC_CREATE_CL_EXCEPTION(invalid_arg_size,                  "invalid arg size");
   ISAAC_CREATE_CL_EXCEPTION(invalid_kernel_args,               "invalid kernel args");
   ISAAC_CREATE_CL_EXCEPTION(invalid_work_dimension,            "invalid work dimension");
   ISAAC_CREATE_CL_EXCEPTION(invalid_work_group_size,           "invalid work group size");
   ISAAC_CREATE_CL_EXCEPTION(invalid_work_item_size,            "invalid work item size");
   ISAAC_CREATE_CL_EXCEPTION(invalid_global_offset,             "invalid global offset");
   ISAAC_CREATE_CL_EXCEPTION(invalid_event_wait_list,           "invalid event wait list");
   ISAAC_CREATE_CL_EXCEPTION(invalid_event,                     "invalid event");
   ISAAC_CREATE_CL_EXCEPTION(invalid_operation,                 "invalid operation");
   ISAAC_CREATE_CL_EXCEPTION(invalid_gl_object,                 "invalid GL object");
   ISAAC_CREATE_CL_EXCEPTION(invalid_buffer_size,               "invalid buffer size");
   ISAAC_CREATE_CL_EXCEPTION(invalid_mip_level,                 "invalid MIP level");
   ISAAC_CREATE_CL_EXCEPTION(invalid_global_work_size,          "invalid global work size");
#ifdef CL_INVALID_PROPERTY
   ISAAC_CREATE_CL_EXCEPTION(invalid_property,                  "invalid property");
#endif
}


}
}

RESTORE_MSVC_WARNING_C4275

#endif
