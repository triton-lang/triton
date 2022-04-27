#pragma once

#ifndef _TRITON_DRIVER_ERROR_H_
#define _TRITON_DRIVER_ERROR_H_

#include <exception>
#include "triton/driver/dispatch.h"


namespace triton
{

  namespace driver
  {

  namespace exception
  {

  namespace nvrtc
  {

#define TRITON_CREATE_NVRTC_EXCEPTION(name, msg) \
class name: public std::exception { public: const char * what() const throw() override { return "NVRTC: Error- " msg; } }

  TRITON_CREATE_NVRTC_EXCEPTION(out_of_memory              ,"out of memory");
  TRITON_CREATE_NVRTC_EXCEPTION(program_creation_failure   ,"program creation failure");
  TRITON_CREATE_NVRTC_EXCEPTION(invalid_input              ,"invalid input");
  TRITON_CREATE_NVRTC_EXCEPTION(invalid_program            ,"invalid program");
  TRITON_CREATE_NVRTC_EXCEPTION(invalid_option             ,"invalid option");
  TRITON_CREATE_NVRTC_EXCEPTION(compilation                ,"compilation");
  TRITON_CREATE_NVRTC_EXCEPTION(builtin_operation_failure  ,"builtin operation failure");
  TRITON_CREATE_NVRTC_EXCEPTION(unknown_error              ,"unknown error");

#undef TRITON_CREATE_NVRTC_EXCEPTION
  }


  namespace cuda
  {
  class base: public std::exception{};

#define TRITON_CREATE_CUDA_EXCEPTION(name, msg) \
class name: public base { public:const char * what() const throw() override { return "CUDA: Error- " msg; } }


  TRITON_CREATE_CUDA_EXCEPTION(invalid_value                   ,"invalid value");
  TRITON_CREATE_CUDA_EXCEPTION(out_of_memory                   ,"out of memory");
  TRITON_CREATE_CUDA_EXCEPTION(not_initialized                 ,"not initialized");
  TRITON_CREATE_CUDA_EXCEPTION(deinitialized                   ,"deinitialized");
  TRITON_CREATE_CUDA_EXCEPTION(profiler_disabled               ,"profiler disabled");
  TRITON_CREATE_CUDA_EXCEPTION(profiler_not_initialized        ,"profiler not initialized");
  TRITON_CREATE_CUDA_EXCEPTION(profiler_already_started        ,"profiler already started");
  TRITON_CREATE_CUDA_EXCEPTION(profiler_already_stopped        ,"profiler already stopped");
  TRITON_CREATE_CUDA_EXCEPTION(no_device                       ,"no device");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_device                  ,"invalid device");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_image                   ,"invalid image");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_context                 ,"invalid context");
  TRITON_CREATE_CUDA_EXCEPTION(context_already_current         ,"context already current");
  TRITON_CREATE_CUDA_EXCEPTION(map_failed                      ,"map failed");
  TRITON_CREATE_CUDA_EXCEPTION(unmap_failed                    ,"unmap failed");
  TRITON_CREATE_CUDA_EXCEPTION(array_is_mapped                 ,"array is mapped");
  TRITON_CREATE_CUDA_EXCEPTION(already_mapped                  ,"already mapped");
  TRITON_CREATE_CUDA_EXCEPTION(no_binary_for_gpu               ,"no binary for gpu");
  TRITON_CREATE_CUDA_EXCEPTION(already_acquired                ,"already acquired");
  TRITON_CREATE_CUDA_EXCEPTION(not_mapped                      ,"not mapped");
  TRITON_CREATE_CUDA_EXCEPTION(not_mapped_as_array             ,"not mapped as array");
  TRITON_CREATE_CUDA_EXCEPTION(not_mapped_as_pointer           ,"not mapped as pointer");
  TRITON_CREATE_CUDA_EXCEPTION(ecc_uncorrectable               ,"ecc uncorrectable");
  TRITON_CREATE_CUDA_EXCEPTION(unsupported_limit               ,"unsupported limit");
  TRITON_CREATE_CUDA_EXCEPTION(context_already_in_use          ,"context already in use");
  TRITON_CREATE_CUDA_EXCEPTION(peer_access_unsupported         ,"peer access unsupported");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_ptx                     ,"invalid ptx");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_graphics_context        ,"invalid graphics context");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_source                  ,"invalid source");
  TRITON_CREATE_CUDA_EXCEPTION(file_not_found                  ,"file not found");
  TRITON_CREATE_CUDA_EXCEPTION(shared_object_symbol_not_found  ,"shared object symbol not found");
  TRITON_CREATE_CUDA_EXCEPTION(shared_object_init_failed       ,"shared object init failed");
  TRITON_CREATE_CUDA_EXCEPTION(operating_system                ,"operating system");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_handle                  ,"invalid handle");
  TRITON_CREATE_CUDA_EXCEPTION(not_found                       ,"not found");
  TRITON_CREATE_CUDA_EXCEPTION(not_ready                       ,"not ready");
  TRITON_CREATE_CUDA_EXCEPTION(illegal_address                 ,"illegal address");
  TRITON_CREATE_CUDA_EXCEPTION(launch_out_of_resources         ,"launch out of resources");
  TRITON_CREATE_CUDA_EXCEPTION(launch_timeout                  ,"launch timeout");
  TRITON_CREATE_CUDA_EXCEPTION(launch_incompatible_texturing   ,"launch incompatible texturing");
  TRITON_CREATE_CUDA_EXCEPTION(peer_access_already_enabled     ,"peer access already enabled");
  TRITON_CREATE_CUDA_EXCEPTION(peer_access_not_enabled         ,"peer access not enabled");
  TRITON_CREATE_CUDA_EXCEPTION(primary_context_active          ,"primary context active");
  TRITON_CREATE_CUDA_EXCEPTION(context_is_destroyed            ,"context is destroyed");
  TRITON_CREATE_CUDA_EXCEPTION(assert_error                    ,"assert");
  TRITON_CREATE_CUDA_EXCEPTION(too_many_peers                  ,"too many peers");
  TRITON_CREATE_CUDA_EXCEPTION(host_memory_already_registered  ,"host memory already registered");
  TRITON_CREATE_CUDA_EXCEPTION(host_memory_not_registered      ,"hot memory not registered");
  TRITON_CREATE_CUDA_EXCEPTION(hardware_stack_error            ,"hardware stack error");
  TRITON_CREATE_CUDA_EXCEPTION(illegal_instruction             ,"illegal instruction");
  TRITON_CREATE_CUDA_EXCEPTION(misaligned_address              ,"misaligned address");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_address_space           ,"invalid address space");
  TRITON_CREATE_CUDA_EXCEPTION(invalid_pc                      ,"invalid pc");
  TRITON_CREATE_CUDA_EXCEPTION(launch_failed                   ,"launch failed");
  TRITON_CREATE_CUDA_EXCEPTION(not_permitted                   ,"not permitted");
  TRITON_CREATE_CUDA_EXCEPTION(not_supported                   ,"not supported");
  TRITON_CREATE_CUDA_EXCEPTION(unknown                         ,"unknown");

#undef TRITON_CREATE_CUDA_EXCEPTION
  }

  namespace cublas
  {
  class base: public std::exception{};

#define TRITON_CREATE_CUBLAS_EXCEPTION(name, msg) \
class name: public base { public: const char * what() const throw() override { return "CUBLAS: Error- " msg; } }

  TRITON_CREATE_CUBLAS_EXCEPTION(not_initialized              ,"not initialized");
  TRITON_CREATE_CUBLAS_EXCEPTION(alloc_failed                 ,"alloc failed");
  TRITON_CREATE_CUBLAS_EXCEPTION(invalid_value                ,"invalid value");
  TRITON_CREATE_CUBLAS_EXCEPTION(arch_mismatch                ,"arch mismatch");
  TRITON_CREATE_CUBLAS_EXCEPTION(mapping_error                ,"mapping error");
  TRITON_CREATE_CUBLAS_EXCEPTION(execution_failed             ,"execution failed");
  TRITON_CREATE_CUBLAS_EXCEPTION(internal_error               ,"internal error");
  TRITON_CREATE_CUBLAS_EXCEPTION(not_supported                ,"not supported");
  TRITON_CREATE_CUBLAS_EXCEPTION(license_error                ,"license error");
  TRITON_CREATE_CUBLAS_EXCEPTION(unknown                      ,"unknown");

#undef TRITON_CREATE_CUBLAS_EXCEPTION
  }

  namespace cudnn
  {
#define TRITON_CREATE_CUDNN_EXCEPTION(name, msg) \
class name: public std::exception { public: const char * what() const throw() override { return "CUDNN: Error- " msg; } }

  TRITON_CREATE_CUDNN_EXCEPTION(not_initialized              ,"not initialized");
  TRITON_CREATE_CUDNN_EXCEPTION(alloc_failed                 ,"allocation failed");
  TRITON_CREATE_CUDNN_EXCEPTION(bad_param                    ,"bad param");
  TRITON_CREATE_CUDNN_EXCEPTION(internal_error               ,"internal error");
  TRITON_CREATE_CUDNN_EXCEPTION(invalid_value                ,"invalid value");
  TRITON_CREATE_CUDNN_EXCEPTION(arch_mismatch                ,"arch mismatch");
  TRITON_CREATE_CUDNN_EXCEPTION(mapping_error                ,"mapping error");
  TRITON_CREATE_CUDNN_EXCEPTION(execution_failed             ,"execution failed");
  TRITON_CREATE_CUDNN_EXCEPTION(not_supported                ,"not supported");
  TRITON_CREATE_CUDNN_EXCEPTION(license_error                ,"license error");
  TRITON_CREATE_CUDNN_EXCEPTION(runtime_prerequisite_missing ,"prerequisite missing");
  TRITON_CREATE_CUDNN_EXCEPTION(runtime_in_progress          ,"runtime in progress");
  TRITON_CREATE_CUDNN_EXCEPTION(runtime_fp_overflow          ,"runtime fp overflow");
  }




  namespace hip
  {
  class base: public std::exception{};

#define TRITON_CREATE_HIP_EXCEPTION(name, msg) \
class name: public base { public:const char * what() const throw() override { return "HIP: Error- " msg; } }


  TRITON_CREATE_HIP_EXCEPTION(invalid_value                   ,"invalid value");
  TRITON_CREATE_HIP_EXCEPTION(out_of_memory                   ,"out of memory");
  TRITON_CREATE_HIP_EXCEPTION(not_initialized                 ,"not initialized");
  TRITON_CREATE_HIP_EXCEPTION(deinitialized                   ,"deinitialized");
  TRITON_CREATE_HIP_EXCEPTION(profiler_disabled               ,"profiler disabled");
  TRITON_CREATE_HIP_EXCEPTION(profiler_not_initialized        ,"profiler not initialized");
  TRITON_CREATE_HIP_EXCEPTION(profiler_already_started        ,"profiler already started");
  TRITON_CREATE_HIP_EXCEPTION(profiler_already_stopped        ,"profiler already stopped");
  TRITON_CREATE_HIP_EXCEPTION(no_device                       ,"no device");
  TRITON_CREATE_HIP_EXCEPTION(invalid_device                  ,"invalid device");
  TRITON_CREATE_HIP_EXCEPTION(invalid_image                   ,"invalid image");
  TRITON_CREATE_HIP_EXCEPTION(invalid_context                 ,"invalid context");
  TRITON_CREATE_HIP_EXCEPTION(context_already_current         ,"context already current");
  TRITON_CREATE_HIP_EXCEPTION(map_failed                      ,"map failed");
  TRITON_CREATE_HIP_EXCEPTION(unmap_failed                    ,"unmap failed");
  TRITON_CREATE_HIP_EXCEPTION(array_is_mapped                 ,"array is mapped");
  TRITON_CREATE_HIP_EXCEPTION(already_mapped                  ,"already mapped");
  TRITON_CREATE_HIP_EXCEPTION(no_binary_for_gpu               ,"no binary for gpu");
  TRITON_CREATE_HIP_EXCEPTION(already_acquired                ,"already acquired");
  TRITON_CREATE_HIP_EXCEPTION(not_mapped                      ,"not mapped");
  TRITON_CREATE_HIP_EXCEPTION(not_mapped_as_array             ,"not mapped as array");
  TRITON_CREATE_HIP_EXCEPTION(not_mapped_as_pointer           ,"not mapped as pointer");
  TRITON_CREATE_HIP_EXCEPTION(ecc_uncorrectable               ,"ecc uncorrectable");
  TRITON_CREATE_HIP_EXCEPTION(unsupported_limit               ,"unsupported limit");
  TRITON_CREATE_HIP_EXCEPTION(context_already_in_use          ,"context already in use");
  TRITON_CREATE_HIP_EXCEPTION(peer_access_unsupported         ,"peer access unsupported");
  TRITON_CREATE_HIP_EXCEPTION(invalid_ptx                     ,"invalid ptx");
  TRITON_CREATE_HIP_EXCEPTION(invalid_graphics_context        ,"invalid graphics context");
  TRITON_CREATE_HIP_EXCEPTION(invalid_source                  ,"invalid source");
  TRITON_CREATE_HIP_EXCEPTION(file_not_found                  ,"file not found");
  TRITON_CREATE_HIP_EXCEPTION(shared_object_symbol_not_found  ,"shared object symbol not found");
  TRITON_CREATE_HIP_EXCEPTION(shared_object_init_failed       ,"shared object init failed");
  TRITON_CREATE_HIP_EXCEPTION(operating_system                ,"operating system");
  TRITON_CREATE_HIP_EXCEPTION(invalid_handle                  ,"invalid handle");
  TRITON_CREATE_HIP_EXCEPTION(not_found                       ,"not found");
  TRITON_CREATE_HIP_EXCEPTION(not_ready                       ,"not ready");
  TRITON_CREATE_HIP_EXCEPTION(illegal_address                 ,"illegal address");
  TRITON_CREATE_HIP_EXCEPTION(launch_out_of_resources         ,"launch out of resources");
  TRITON_CREATE_HIP_EXCEPTION(launch_timeout                  ,"launch timeout");
  TRITON_CREATE_HIP_EXCEPTION(launch_incompatible_texturing   ,"launch incompatible texturing");
  TRITON_CREATE_HIP_EXCEPTION(peer_access_already_enabled     ,"peer access already enabled");
  TRITON_CREATE_HIP_EXCEPTION(peer_access_not_enabled         ,"peer access not enabled");
  TRITON_CREATE_HIP_EXCEPTION(primary_context_active          ,"primary context active");
  TRITON_CREATE_HIP_EXCEPTION(context_is_destroyed            ,"context is destroyed");
  TRITON_CREATE_HIP_EXCEPTION(assert_error                    ,"assert");
  TRITON_CREATE_HIP_EXCEPTION(too_many_peers                  ,"too many peers");
  TRITON_CREATE_HIP_EXCEPTION(host_memory_already_registered  ,"host memory already registered");
  TRITON_CREATE_HIP_EXCEPTION(host_memory_not_registered      ,"hot memory not registered");
  TRITON_CREATE_HIP_EXCEPTION(hardware_stack_error            ,"hardware stack error");
  TRITON_CREATE_HIP_EXCEPTION(illegal_instruction             ,"illegal instruction");
  TRITON_CREATE_HIP_EXCEPTION(misaligned_address              ,"misaligned address");
  TRITON_CREATE_HIP_EXCEPTION(invalid_address_space           ,"invalid address space");
  TRITON_CREATE_HIP_EXCEPTION(invalid_pc                      ,"invalid pc");
  TRITON_CREATE_HIP_EXCEPTION(launch_failed                   ,"launch failed");
  TRITON_CREATE_HIP_EXCEPTION(not_permitted                   ,"not permitted");
  TRITON_CREATE_HIP_EXCEPTION(not_supported                   ,"not supported");
  TRITON_CREATE_HIP_EXCEPTION(invalid_symbol                   ,"invalid symbol");
  TRITON_CREATE_HIP_EXCEPTION(unknown                         ,"unknown");

#undef TRITON_CREATE_CUDA_EXCEPTION
  }

  }
  }
}

#endif
