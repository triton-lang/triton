find_path(OPENBLAS_INCLUDE_DIR cblas.h)
find_library(OPENBLAS_LIBRARIES NAMES openblas PATHS /lib/ /lib64/  /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/OpenBLAS/lib $ENV{OPENBLAS_HOME}/lib)

if(OPENBLAS_LIBRARIES)
    set(OPENBLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenBlas DEFAULT_MSG OPENBLAS_LIBRARIES OPENBLAS_INCLUDE_DIR)
mark_as_advanced(OpenBlas)
