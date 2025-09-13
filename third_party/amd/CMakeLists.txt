include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

find_program(HIPCONFIG_EXECUTABLE hipconfig)
if(HIPCONFIG_EXECUTABLE)
    execute_process(COMMAND ${HIPCONFIG_EXECUTABLE} --path
                   OUTPUT_VARIABLE ROCM_PATH
                   OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Found ROCm at: ${ROCM_PATH}")
endif()

if(ROCM_PATH AND EXISTS "${ROCM_PATH}")
    set(ROCM_INCLUDE_DIR "${ROCM_PATH}/include")
    include_directories("${ROCM_INCLUDE_DIR}/hipblas-common")
    message(STATUS "Added ROCm include directories: ${ROCM_INCLUDE_DIR}")
else()
    # error out maybe?
    message(WARNING "ROCm installation not found. HIP BLAS headers may not be available.")
endif()

add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
  include_directories(${LLD_INCLUDE_DIRS})
  message(STATUS "Found LLD distro-package @ ${LLD_DIR} and LLD include dirs @ ${LLD_INCLUDE_DIRS}")
  add_triton_plugin(TritonAMD ${CMAKE_CURRENT_SOURCE_DIR}/python/triton_amd.cc LINK_LIBS TritonAMDGPUToLLVM TritonAMDGPUTransforms TritonAMDGPUDialectToLLVM)
  target_link_libraries(TritonAMD PRIVATE Python3::Module pybind11::headers lldCommon lldELF)
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
add_subdirectory(test)
