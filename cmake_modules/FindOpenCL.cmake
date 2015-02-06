file(GLOB AMDAPPSDK_ROOT /opt/AMDAPPSDK*)

find_package(CUDA QUIET)
find_library(OPENCL_LIBRARIES NAMES OpenCL HINTS ${AMDAPPSDK_ROOT}/lib/x86_64/ ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL  DEFAULT_MSG OPENCL_LIBRARIES)
mark_as_advanced(OpenCL)
