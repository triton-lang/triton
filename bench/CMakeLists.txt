set(CMAKE_BUILD_TYPE Release)

set(BLAS_DEF)
set(BLAS_LIBS)

#CUBLAS
find_package(CUDA)
if(CUDA_FOUND)
    set(BLAS_DEF ${BLAS_DEF} "-DHAS_A_BLAS -DBENCH_CUBLAS")
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

#CLAMDBLAS
find_package(CLAMDBLAS)
if(CLAMDBLAS_FOUND)
    set(BLAS_DEF ${BLAS_DEF} "-DHAS_A_BLAS -DBENCH_CLBLAS")
    include_directories(${CLAMDBLAS_INCLUDE_DIR})
    set(BLAS_LIBS ${BLAS_LIBS}  ${CLAMDBLAS_LIBRARIES} )
endif()

#CBLAS
find_package(MKL)
if(MKL_FOUND)
    set(BLAS_DEF ${BLAS_DEF} "-DHAS_A_BLAS -DBENCH_CBLAS")
    include_directories(${MKL_INCLUDE_DIR})
    set(BLAS_LIBS ${BLAS_LIBS}  ${MKL_LIBRARIES} )
else()

#    find_package(OpenBlas)
#    if(OPENBLAS_FOUND)
#        set(BLAS_DEF ${BLAS_DEF} "-DBENCH_CBLAS")
#        include_directories(${OPENBLAS_INCLUDE_DIR})
#        set(BLAS_LIBS ${BLAS_LIBS}  ${OPENBLAS_LIBRARIES} )
#    endif()
endif()

string(REPLACE ";" " " BLAS_DEF_STR "${BLAS_DEF}")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
foreach(PROG blas)
   if(CUDA_FOUND)
       set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} " ${BLAS_DEF_STR} -std=c++11 ${BACKEND_DEFINES}")
       set(CUPROG ${CMAKE_CURRENT_BINARY_DIR}/${PROG}.cu)
       file(COPY ${PROG}.cpp DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
       file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/${PROG}.cpp ${CUPROG})
       cuda_add_executable(${PROG}-bench  ${CUPROG})
      cuda_add_cublas_to_target(${PROG}-bench)
   else()
       add_executable(${PROG}-bench  ${PROG}.cpp)
       set_target_properties(${PROG}-bench PROPERTIES COMPILE_FLAGS "${BLAS_DEF_STR}")
   endif()
   target_link_libraries(${PROG}-bench ${BLAS_LIBS} isaac ${OPENCL_LIBRARIES} )
endforeach(PROG)
