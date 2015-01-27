set(CMAKE_BUILD_TYPE Release)

set(BLAS_DEF)
set(BLAS_LIBS atidlas)

#CUBLAS
find_package(CUDA)
if(CUDA_FOUND)
    set(BLAS_DEF ${BLAS_DEF} "-DBENCH_CUBLAS")
endif()

#CLAMDBLAS
find_package(CLAMDBLAS)
if(CLAMDBLAS_FOUND)
    set(BLAS_DEF ${BLAS_DEF} "-DBENCH_CLAMDBLAS")
    include_directories(${CLAMDBLAS_INCLUDE_DIR})
    set(BLAS_LIBS ${BLAS_LIBS}  ${CLAMDBLAS_LIBRARIES} )
endif()

#CBLAS
find_package(MKL)
if(MKL_FOUND)
    set(BLAS_DEF ${BLAS_DEF} "-DBENCH_CBLAS")
    include_directories(${MKL_INCLUDE_DIR})
    set(BLAS_LIBS ${BLAS_LIBS}  ${MKL_LIBRARIES} )
endif()

string(REPLACE ";" " " BLAS_DEF_STR "${BLAS_DEF}")

foreach(PROG blas overhead)
     add_executable(${PROG}-bench  ${PROG}.cpp)
     target_link_libraries(${PROG}-bench ${BLAS_LIBS} ${OPENCL_LIBRARIES})
     set_target_properties(${PROG}-bench PROPERTIES COMPILE_FLAGS "-Wall -Wextra ${BLAS_DEF_STR}")
endforeach(PROG)
