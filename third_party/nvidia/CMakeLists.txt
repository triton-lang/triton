include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  set(GSAN_RUNTIME_SRC "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/GSanLibrary.cu")
  set(GSAN_RUNTIME_HDRS
    "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/GSan.h"
    "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src/Hash.cuh")
  set(GSAN_RUNTIME_IR "${CMAKE_CURRENT_SOURCE_DIR}/backend/lib/gsan.ll")

  find_program(TRITON_GSAN_CLANGXX NAMES clang++
               PATHS "${LLVM_SYSPATH}/bin" NO_DEFAULT_PATH)
  if(NOT TRITON_GSAN_CLANGXX)
    find_program(TRITON_GSAN_CLANGXX NAMES clang++)
  endif()
  if(NOT TRITON_GSAN_CLANGXX)
    message(FATAL_ERROR "clang++ is required to build gsan.ll")
  endif()

  set(GSAN_RUNTIME_STDLIB_INCLUDE_FLAGS)
  foreach(dir IN LISTS CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES)
    list(APPEND GSAN_RUNTIME_STDLIB_INCLUDE_FLAGS -isystem "${dir}")
  endforeach()
  if(NOT GSAN_RUNTIME_STDLIB_INCLUDE_FLAGS)
    message(FATAL_ERROR
      "Failed to determine C++ standard library include directories for gsan.ll")
  endif()

  set(GSAN_RUNTIME_COMPAT_DEFINES)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
    # glibc's aarch64 math-vector declarations reference SVE types that are not
    # available during NVPTX device-only compilation. These typedefs are only
    # used in declarations, so placeholder scalar types are sufficient here.
    list(APPEND GSAN_RUNTIME_COMPAT_DEFINES
         -D__SVFloat32_t=float
         -D__SVFloat64_t=double
         -D__SVBool_t=bool)
  endif()

  add_custom_command(
      OUTPUT "${GSAN_RUNTIME_IR}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${PROJECT_SOURCE_DIR}/third_party/nvidia/backend/lib"
      COMMAND "${TRITON_GSAN_CLANGXX}" -x c++ -std=c++17 -O3 -S -emit-llvm
              -x cuda --cuda-device-only
              -nocudalib
              --no-cuda-version-check
              -fno-exceptions
              -fcuda-flush-denormals-to-zero
              --cuda-gpu-arch=sm_80
              -Wno-unknown-cuda-version
              ${GSAN_RUNTIME_COMPAT_DEFINES}
              ${GSAN_RUNTIME_STDLIB_INCLUDE_FLAGS}
              -isystem "${CMAKE_CURRENT_SOURCE_DIR}/backend/include"
              "${GSAN_RUNTIME_SRC}" -o "${GSAN_RUNTIME_IR}"
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
