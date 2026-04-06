include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  set(GSAN_RUNTIME_SRC "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/GSanLibrary.cu")
  set(GSAN_RUNTIME_HDRS
    "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/GSan.h"
    "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/Hash.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/clang_cuda_shims/assert.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/clang_cuda_shims/curand_mtgp32_kernel.h")
  set(GSAN_RUNTIME_IR "${CMAKE_CURRENT_SOURCE_DIR}/backend/lib/gsan.ll")

  find_program(TRITON_GSAN_CLANGXX NAMES clang++
               PATHS "${LLVM_SYSPATH}/bin"
               REQUIRED NO_DEFAULT_PATH)
  if(NOT TRITON_GSAN_CLANGXX)
    message(FATAL_ERROR "clang++ is required to build gsan.ll")
  endif()

  set(GSAN_RUNTIME_PLATFORM_FLAGS
      "--cuda-path=${CMAKE_CURRENT_SOURCE_DIR}/backend")
  if(APPLE)
    list(APPEND GSAN_RUNTIME_PLATFORM_FLAGS -isysroot "${CMAKE_OSX_SYSROOT}")
  endif()

  add_custom_command(
      OUTPUT "${GSAN_RUNTIME_IR}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${PROJECT_SOURCE_DIR}/third_party/nvidia/backend/lib"
      COMMAND "${TRITON_GSAN_CLANGXX}"
              -x cuda
              -std=c++17 -O3 -S -emit-llvm
              --cuda-device-only
              -nocudalib
              --no-cuda-version-check
              -fno-exceptions
              -fcuda-flush-denormals-to-zero
              --cuda-gpu-arch=sm_80
              -Wno-unknown-cuda-version
              ${GSAN_RUNTIME_PLATFORM_FLAGS}
              -isystem "${CMAKE_CURRENT_SOURCE_DIR}/clang_cuda_shims"
              -isystem "${CMAKE_CURRENT_SOURCE_DIR}/backend/include"
              "${GSAN_RUNTIME_SRC}" -o "${GSAN_RUNTIME_IR}"
      DEPENDS "${GSAN_RUNTIME_SRC}" ${GSAN_RUNTIME_HDRS}
      COMMENT "Building GSan runtime"
      VERBATIM)
  add_custom_target(TritonNVIDIAGSanRuntime ALL DEPENDS "${GSAN_RUNTIME_IR}")

  add_triton_plugin(TritonNVIDIA ${CMAKE_CURRENT_SOURCE_DIR}/triton_nvidia.cc LINK_LIBS TritonNVIDIAGPUToLLVM NVGPUToLLVM)
  target_link_libraries(TritonNVIDIA PRIVATE Python3::Module pybind11::headers)
  add_dependencies(TritonNVIDIA TritonNVIDIAGSanRuntime)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
add_subdirectory(hopper)
