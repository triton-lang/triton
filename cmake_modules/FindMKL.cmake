file(GLOB SYSTEM_STUDIO_ROOT /opt/intel/system_studio_*)

find_path(MKL_INCLUDE_DIR mkl_blas.h HINTS ${SYSTEM_STUDIO_ROOT}/mkl/include/)
find_library(MKL_LIBRARIES NAMES mkl_core HINTS ${SYSTEM_STUDIO_ROOT}/mkl/lib/intel64/)
find_library(ICC_LIBRARIES NAMES iomp5 HINTS ${SYSTEM_STUDIO_ROOT}/compiler/lib/intel64/)

if(ICC_LIBRARIES)
    set(OMP_LIBRARIES ${ICC_LIBRARIES})
else()
    set(OMP_LIBRARIES gomp)
endif()

if(MKL_LIBRARIES AND OMP_LIBRARIES)
    set(MKL_LIBRARIES mkl_intel_lp64 mkl_avx mkl_intel_thread ${MKL_LIBRARIES} ${OMP_LIBRARIES} pthread)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL  DEFAULT_MSG MKL_LIBRARIES MKL_INCLUDE_DIR)
mark_as_advanced(MKL)
