set(CMAKE_BUILD_TYPE Release)

set(BLAS_DEF)
set(BLAS_LIBS atidlas)

#CUBLAS
find_package(CUDA)
if(CUDA_FOUND)
  set(BLAS_DEF ${BLAS_DEF} "-DBENCH_CUBLAS")
endif()
#CLAMDBLAS
include_directories(/opt/clAmdBlas-1.10.321/include/)
set(BLAS_DEF ${BLAS_DEF} "-DBENCH_CLAMDBLAS")
set(BLAS_LIBS ${BLAS_LIBS} /opt/clAmdBlas-1.10.321/lib64/libclAmdBlas.so)
#CBLAS
include_directories(/opt/intel/system_studio_2015.1.045/mkl/include/)
set(BLAS_DEF ${BLAS_DEF} "-DBENCH_CBLAS")
set(MKLROOT /opt/intel/system_studio_2015.1.045/mkl/)
set(CCROOT /opt/intel/system_studio_2015.1.045/compiler/)
set(BLAS_LIBS ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${BLAS_LIBS} ${MKLROOT}/lib/intel64/libmkl_core.a ${CCROOT}/lib/intel64/libiomp5.so pthread dl m)


string(REPLACE ";" " " BLAS_DEF_STR "${BLAS_DEF}")

foreach(PROG blas overhead)
     add_executable(${PROG}-bench  ${PROG}.cpp)
     target_link_libraries(${PROG}-bench ${BLAS_LIBS} ${OPENCL_LIBRARIES})
     set_target_properties(${PROG}-bench PROPERTIES COMPILE_FLAGS "-Wall -Wextra ${BLAS_DEF_STR}")
endforeach(PROG)
