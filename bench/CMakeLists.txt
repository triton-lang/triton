set(CMAKE_BUILD_TYPE Release)

set(BLAS_DEF)
set(BLAS_LIBS)

#CUBLAS
find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(BLAS_DEF ${BLAS_DEF} "-DHAS_A_BLAS -DBENCH_CUBLAS")
    include_directories(${CUDA_INCLUDE_DIRS})
    set(BLAS_LIBS ${BLAS_LIBS} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

#CLBLAS
find_package(CLBLAS QUIET)
if(CLBLAS_FOUND)
    set(BLAS_DEF ${BLAS_DEF} "-DHAS_A_BLAS -DBENCH_CLBLAS")
    include_directories(${CLBLAS_INCLUDE_DIR})
    set(BLAS_LIBS ${BLAS_LIBS}  ${CLBLAS_LIBRARIES} OpenCL pthread)
endif()

##CBLAS
#find_package(MKL QUIET)
#if(MKL_FOUND)
#    set(BLAS_DEF ${BLAS_DEF} "-DHAS_A_BLAS -DBENCH_MKL")
#    include_directories(${MKL_INCLUDE_DIR})
#    set(BLAS_LIBS ${BLAS_LIBS}  ${MKL_LIBRARIES} )
#else()
# find_package(OpenBlas)
# if(OPENBLAS_FOUND)
#     set(BLAS_DEF ${BLAS_DEF} "-DHAS_A_BLAS -DBENCH_CBLAS")
#     include_directories(${OPENBLAS_INCLUDE_DIR})
#     set(BLAS_LIBS ${BLAS_LIBS}  ${OPENBLAS_LIBRARIES} )
# endif()
#endif()

string(REPLACE ";" " " BLAS_DEF_STR "${BLAS_DEF}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tests/common)
foreach(PROG blas)
   add_executable(bench-${PROG}  ${PROG}.cpp)
   set_target_properties(bench-${PROG} PROPERTIES COMPILE_FLAGS "${BLAS_DEF_STR}")
   target_link_libraries(bench-${PROG} ${BLAS_LIBS} isaac)
endforeach(PROG)
