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

  # FIXME: Shouldn't need to manually specify c++ stdlib path
  # FIXME: Not compiling as cuda
  #             -x cuda --cuda-device-only --no-cuda-version-check
  add_custom_command(
      OUTPUT "${GSAN_RUNTIME_IR}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
              "${PROJECT_SOURCE_DIR}/third_party/nvidia/backend/lib"
      COMMAND "${TRITON_GSAN_CLANGXX}" -x c++ -std=c++17 -O3 -S -emit-llvm
              -fno-exceptions
              -Wno-unknown-cuda-version
              -isystem /usr/include/c++/13/
              -isystem /usr/include/aarch64-linux-gnu/c++/13
              -isystem "${CMAKE_CURRENT_SOURCE_DIR}/backend/include"
              "${GSAN_RUNTIME_SRC}" -o "${GSAN_RUNTIME_IR}"
      DEPENDS "${GSAN_RUNTIME_SRC}" "${GSAN_RUNTIME_HDR}"
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
