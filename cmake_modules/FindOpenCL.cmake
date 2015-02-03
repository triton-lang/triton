file(GLOB AMDAPPSDK_ROOT /opt/AMDAPPSDK*)

find_package(CUDA QUIET)
find_path(OPENCL_INCLUDE_DIR CL/cl.hpp HINTS ${AMDAPPSDK_ROOT}/include/ ${CUDA_SDK_ROOT_DIR}/include)
find_library(OPENCL_LIBRARIES NAMES OpenCL HINTS ${AMDAPPSDK_ROOT}/lib/x86_64/ ${CUDA_SDK_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL  DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIR)
mark_as_advanced(OpenCL)
