include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  if(TRITON_BUILD_NVIDIA_GSAN_RUNTIME)
    set(GSAN_RUNTIME_DIR "${triton_SOURCE_DIR}/python/triton/experimental/gsan/src")
    set(GSAN_RUNTIME_HDRS
      "${GSAN_RUNTIME_DIR}/GSan.h"
      "${GSAN_RUNTIME_DIR}/Compat.h"
      "${GSAN_RUNTIME_DIR}/TMA.cuh"
      "${GSAN_RUNTIME_DIR}/AllocationProfile.h"
      "${GSAN_RUNTIME_DIR}/Hash.cuh")

    find_program(TRITON_GSAN_CLANGXX NAMES clang++
                 PATHS "${LLVM_SYSPATH}/bin"
                 REQUIRED NO_DEFAULT_PATH)
    if(NOT TRITON_GSAN_CLANGXX)
      message(FATAL_ERROR "clang++ is required to build gsan.ll")
    endif()

    set(TRITON_GSAN_INCLUDE_FLAGS "")
    foreach(include_path IN ITEMS "${TRITON_CUDACRT_PATH}" "${TRITON_CUDART_PATH}")
      if(NOT "${include_path}" STREQUAL "")
        list(APPEND TRITON_GSAN_INCLUDE_FLAGS "-I${include_path}")
      endif()
    endforeach()

    # Keep both runtimes on the same compiler flags and dependency set.
    # They share the instrumentation ABI and descriptor decoder.
    set(GSAN_RUNTIME_IRS "")
    foreach(runtime IN ITEMS gsan gsan-profile)
      if(runtime STREQUAL "gsan-profile")
        set(runtime_src "${GSAN_RUNTIME_DIR}/AllocationProfile.cu")
      else()
        set(runtime_src "${GSAN_RUNTIME_DIR}/GSanLibrary.cu")
      endif()
      set(runtime_ir "${CMAKE_CURRENT_SOURCE_DIR}/backend/lib/${runtime}.ll")
      add_custom_command(
          OUTPUT "${runtime_ir}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory
                  "${CMAKE_CURRENT_SOURCE_DIR}/backend/lib"
          COMMAND "${TRITON_GSAN_CLANGXX}"
                  -x cuda -std=c++17 -O3 -S -emit-llvm --cuda-device-only
                  -nocudainc -nocudalib -nostdinc -fno-exceptions
                  -fcuda-flush-denormals-to-zero --cuda-gpu-arch=sm_80
                  --cuda-feature=+ptx70
                  ${TRITON_GSAN_INCLUDE_FLAGS}
                  "${runtime_src}" -o "${runtime_ir}"
          DEPENDS "${runtime_src}" ${GSAN_RUNTIME_HDRS}
          COMMENT "Building ${runtime} runtime"
          VERBATIM)
      list(APPEND GSAN_RUNTIME_IRS "${runtime_ir}")
    endforeach()
    add_custom_target(TritonNVIDIAGSanRuntime ALL DEPENDS ${GSAN_RUNTIME_IRS})
  endif()

  add_triton_plugin(TritonNVIDIA ${CMAKE_CURRENT_SOURCE_DIR}/triton_nvidia.cc LINK_LIBS TritonNVIDIAGPUToLLVM NVGPUToLLVM)
  target_link_libraries(TritonNVIDIA PRIVATE Python3::Module nanobind-static)
  if(TRITON_BUILD_NVIDIA_GSAN_RUNTIME)
    add_dependencies(TritonNVIDIA TritonNVIDIAGSanRuntime)
  endif()
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
add_subdirectory(hopper)
