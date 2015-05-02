if(ANDROID)
    file(GLOB ADRENO_SDK_ROOT /opt/adreno-sdk*)
    set(L_HINTS ${ADRENO_SDK_ROOT}/driver/lib/)
else()
    file(GLOB AMDAPPSDK_ROOT /opt/AMDAPPSDK*)
    set(L_HINTS ${AMDAPPSDK_ROOT}/lib/x86_64/ ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/)
endif()

find_library(OPENCL_LIBRARIES NAMES OpenCL NO_CMAKE_FIND_ROOT_PATH HINTS ${L_HINTS} )
message(STATUS ${OPENCL_LIBRARIES})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL  DEFAULT_MSG OPENCL_LIBRARIES)
mark_as_advanced(OpenCL)
