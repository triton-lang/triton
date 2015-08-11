#Hints for finding libOpenCL

#OpenCL Hints
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(L_HINTS $ENV{INTELOCLSDKROOT}/lib/x64)
else()
    set(L_HINTS $ENV{INTELOCLSDKROOT}/lib/x86)
endif()

if(ANDROID)
    set(ANDROID_CL_GLOB_HINTS /opt/adreno-driver*/lib)
    foreach(PATH ${ANDROID_CL_GLOB_HINTS})
        file(GLOB _TMP ${PATH})
        set(L_HINTS ${L_HINTS} ${_TMP})
    endforeach()
else()
    set(X86_CL_GLOB_HINTS /opt/AMDAPPSDK*/lib/x86_64)
    foreach(PATH ${X86_CL_GLOB_HINTS})
        file(GLOB _TMP ${PATH})
        set(L_HINTS ${L_HINTS} ${_TMP})
    endforeach()
    set(L_HINTS ${L_HINTS} ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/)
endif()

find_library(OPENCL_LIBRARIES NAMES OpenCL HINTS ${L_HINTS} )
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL DEFAULT_MSG OPENCL_LIBRARIES)
mark_as_advanced(OpenCL)
