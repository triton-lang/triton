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

#include "triton/driver/error.h"

namespace triton {
namespace driver {

void check(CUresult err) {
  using namespace exception::cuda;
  switch (err) {
  case CUDA_SUCCESS:
    break;
  case CUDA_ERROR_INVALID_VALUE:
    throw invalid_value();
  case CUDA_ERROR_OUT_OF_MEMORY:
    throw out_of_memory();
  case CUDA_ERROR_NOT_INITIALIZED:
    throw not_initialized();
  case CUDA_ERROR_DEINITIALIZED:
    throw deinitialized();
  case CUDA_ERROR_PROFILER_DISABLED:
    throw profiler_disabled();
  case CUDA_ERROR_PROFILER_NOT_INITIALIZED:
    throw profiler_not_initialized();
  case CUDA_ERROR_PROFILER_ALREADY_STARTED:
    throw profiler_already_started();
  case CUDA_ERROR_PROFILER_ALREADY_STOPPED:
    throw profiler_already_stopped();
  case CUDA_ERROR_NO_DEVICE:
    throw no_device();
  case CUDA_ERROR_INVALID_DEVICE:
    throw invalid_device();
  case CUDA_ERROR_INVALID_IMAGE:
    throw invalid_image();
  case CUDA_ERROR_INVALID_CONTEXT:
    throw invalid_context();
  case CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
    throw context_already_current();
  case CUDA_ERROR_MAP_FAILED:
    throw map_failed();
  case CUDA_ERROR_UNMAP_FAILED:
    throw unmap_failed();
  case CUDA_ERROR_ARRAY_IS_MAPPED:
    throw array_is_mapped();
  case CUDA_ERROR_ALREADY_MAPPED:
    throw already_mapped();
  case CUDA_ERROR_NO_BINARY_FOR_GPU:
    throw no_binary_for_gpu();
  case CUDA_ERROR_ALREADY_ACQUIRED:
    throw already_acquired();
  case CUDA_ERROR_NOT_MAPPED:
    throw not_mapped();
  case CUDA_ERROR_NOT_MAPPED_AS_ARRAY:
    throw not_mapped_as_array();
  case CUDA_ERROR_NOT_MAPPED_AS_POINTER:
    throw not_mapped_as_pointer();
  case CUDA_ERROR_ECC_UNCORRECTABLE:
    throw ecc_uncorrectable();
  case CUDA_ERROR_UNSUPPORTED_LIMIT:
    throw unsupported_limit();
  case CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
    throw context_already_in_use();
  case CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
    throw peer_access_unsupported();
  case CUDA_ERROR_INVALID_PTX:
    throw invalid_ptx();
  case CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
    throw invalid_graphics_context();
  case CUDA_ERROR_INVALID_SOURCE:
    throw invalid_source();
  case CUDA_ERROR_FILE_NOT_FOUND:
    throw file_not_found();
  case CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
    throw shared_object_symbol_not_found();
  case CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
    throw shared_object_init_failed();
  case CUDA_ERROR_OPERATING_SYSTEM:
    throw operating_system();
  case CUDA_ERROR_INVALID_HANDLE:
    throw invalid_handle();
  case CUDA_ERROR_NOT_FOUND:
    throw not_found();
  case CUDA_ERROR_NOT_READY:
    throw not_ready();
  case CUDA_ERROR_ILLEGAL_ADDRESS:
    throw illegal_address();
  case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
    throw launch_out_of_resources();
  case CUDA_ERROR_LAUNCH_TIMEOUT:
    throw launch_timeout();
  case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
    throw launch_incompatible_texturing();
  case CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
    throw peer_access_already_enabled();
  case CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
    throw peer_access_not_enabled();
  case CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
    throw primary_context_active();
  case CUDA_ERROR_CONTEXT_IS_DESTROYED:
    throw context_is_destroyed();
  case CUDA_ERROR_ASSERT:
    throw assert_error();
  case CUDA_ERROR_TOO_MANY_PEERS:
    throw too_many_peers();
  case CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
    throw host_memory_already_registered();
  case CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
    throw host_memory_not_registered();
  case CUDA_ERROR_HARDWARE_STACK_ERROR:
    throw hardware_stack_error();
  case CUDA_ERROR_ILLEGAL_INSTRUCTION:
    throw illegal_instruction();
  case CUDA_ERROR_MISALIGNED_ADDRESS:
    throw misaligned_address();
  case CUDA_ERROR_INVALID_ADDRESS_SPACE:
    throw invalid_address_space();
  case CUDA_ERROR_INVALID_PC:
    throw invalid_pc();
  case CUDA_ERROR_LAUNCH_FAILED:
    throw launch_failed();
  case CUDA_ERROR_NOT_PERMITTED:
    throw not_permitted();
  case CUDA_ERROR_NOT_SUPPORTED:
    throw not_supported();
  case CUDA_ERROR_UNKNOWN:
    throw unknown();
  default:
    throw unknown();
  }
}

void check(hipError_t error) {
  using namespace exception::hip;
  switch (error) {
  case hipSuccess:
    break;
  case hipErrorInvalidValue:
    throw invalid_value();
  case hipErrorMemoryAllocation:
    throw out_of_memory();
  case hipErrorNotInitialized:
    throw not_initialized();
  case hipErrorDeinitialized:
    throw deinitialized();
  case hipErrorProfilerDisabled:
    throw profiler_disabled();
  case hipErrorProfilerNotInitialized:
    throw profiler_not_initialized();
  case hipErrorProfilerAlreadyStarted:
    throw profiler_already_started();
  case hipErrorProfilerAlreadyStopped:
    throw profiler_already_stopped();
  case hipErrorNoDevice:
    throw no_device();
  case hipErrorInvalidSymbol:
    throw invalid_symbol();
  case hipErrorInvalidDevice:
    throw invalid_device();
  case hipErrorInvalidImage:
    throw invalid_image();
  case hipErrorInvalidContext:
    throw invalid_context();
  case hipErrorContextAlreadyCurrent:
    throw context_already_current();
  case hipErrorMapFailed:
    throw map_failed();
  case hipErrorUnmapFailed:
    throw unmap_failed();
  case hipErrorArrayIsMapped:
    throw array_is_mapped();
  case hipErrorAlreadyMapped:
    throw already_mapped();
  case hipErrorNoBinaryForGpu:
    throw no_binary_for_gpu();
  case hipErrorAlreadyAcquired:
    throw already_acquired();
  case hipErrorNotMapped:
    throw not_mapped();
  case hipErrorNotMappedAsArray:
    throw not_mapped_as_array();
  case hipErrorNotMappedAsPointer:
    throw not_mapped_as_pointer();
  case hipErrorECCNotCorrectable:
    throw ecc_uncorrectable();
  case hipErrorUnsupportedLimit:
    throw unsupported_limit();
  case hipErrorContextAlreadyInUse:
    throw context_already_in_use();
  case hipErrorPeerAccessUnsupported:
    throw peer_access_unsupported();
  case hipErrorInvalidKernelFile:
    throw invalid_ptx();
  case hipErrorInvalidGraphicsContext:
    throw invalid_graphics_context();
  case hipErrorInvalidSource:
    throw invalid_source();
  case hipErrorFileNotFound:
    throw file_not_found();
  case hipErrorSharedObjectSymbolNotFound:
    throw shared_object_symbol_not_found();
  case hipErrorSharedObjectInitFailed:
    throw shared_object_init_failed();
  case hipErrorOperatingSystem:
    throw operating_system();
  case hipErrorInvalidResourceHandle:
    throw invalid_handle();
  case hipErrorNotFound:
    throw not_found();
  case hipErrorNotReady:
    throw not_ready();
  case hipErrorIllegalAddress:
    throw illegal_address();
  case hipErrorLaunchOutOfResources:
    throw launch_out_of_resources();
  case hipErrorLaunchTimeOut:
    throw launch_timeout();
  // case hipErrorLaunchIncompatibleTexturing  : throw
  // launch_incompatible_texturing();
  case hipErrorPeerAccessAlreadyEnabled:
    throw peer_access_already_enabled();
  case hipErrorPeerAccessNotEnabled:
    throw peer_access_not_enabled();
  // case hipErrorPrimaryContextActive         : throw primary_context_active();
  // case hipErrorContextIsDestroyed           : throw context_is_destroyed();
  case hipErrorAssert:
    throw assert_error();
  // case hipErrorTooManyPeers                 : throw too_many_peers();
  case hipErrorHostMemoryAlreadyRegistered:
    throw host_memory_already_registered();
  case hipErrorHostMemoryNotRegistered:
    throw host_memory_not_registered();
  // case hipErrorHardwareStackError           : throw hardware_stack_error();
  // case hipErrorIllegalInstruction            : throw illegal_instruction();
  // case hipErrorMisalignedAddress             : throw misaligned_address();
  // case hipErrorInvalidAddressSpace          : throw invalid_address_space();
  // case hipErrorInvalidPc                     : throw invalid_pc();
  case hipErrorLaunchFailure:
    throw launch_failed();
  // case hipErrorNotPermitted                  : throw not_permitted();
  case hipErrorNotSupported:
    throw not_supported();
  case hipErrorUnknown:
    throw unknown();
  default:
    throw unknown();
  }
}

} // namespace driver
} // namespace triton
