#ifndef ISAAC_DRIVER_COMMON_H
#define ISAAC_DRIVER_COMMON_H

#include <CL/cl.hpp>
#ifdef ISAAC_WITH_CUDA
#include <cuda.h>
#include <nvrtc.h>
#endif

namespace isaac
{
namespace driver
{

enum backend_type
{
  OPENCL
#ifdef ISAAC_WITH_CUDA
  ,CUDA
#endif
};

enum device_type
{
  DEVICE_TYPE_GPU = CL_DEVICE_TYPE_GPU,
  DEVICE_TYPE_CPU = CL_DEVICE_TYPE_CPU,
  DEVICE_TYPE_ACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR
};

#ifdef ISAAC_WITH_CUDA

namespace nvrtc
{
  namespace exception
  {

  #define ISAAC_CREATE_NVRTC_EXCEPTION(name, msg) class name: public std::exception { const char * what() const throw(){ return "NVRTC: Error- " msg; } }

  ISAAC_CREATE_NVRTC_EXCEPTION(out_of_memory              ,"out of memory exception");
  ISAAC_CREATE_NVRTC_EXCEPTION(program_creation_failure   ,"program creation failure");
  ISAAC_CREATE_NVRTC_EXCEPTION(invalid_input              ,"invalid input");
  ISAAC_CREATE_NVRTC_EXCEPTION(invalid_program            ,"invalid program");
  ISAAC_CREATE_NVRTC_EXCEPTION(invalid_option             ,"invalid option");
  ISAAC_CREATE_NVRTC_EXCEPTION(compilation                ,"compilation");
  ISAAC_CREATE_NVRTC_EXCEPTION(builtin_operation_failure  ,"builtin operation failure");
  ISAAC_CREATE_NVRTC_EXCEPTION(unknown_error              ,"unknown error");

  #undef ISAAC_CREATE_NVRTC_EXCEPTION
  }

void check(nvrtcResult err);
}


namespace cuda
{
  namespace exception
  {

    class base: public std::exception{};

    #define ISAAC_CREATE_CUDA_EXCEPTION(name, msg) class name: public base { const char * what() const throw(){ return "CUDA: Error- " msg; } }


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

void check(CUresult);
}

#endif

}
}

#endif
