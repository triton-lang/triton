foreach(PROG blas)
     add_executable(${PROG}-bench ${PROG}.cpp)
     target_link_libraries(${PROG}-bench ${OPENCL_LIBRARIES})
     set_target_properties(${PROG}-bench PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL -Wall -Wextra")
endforeach(PROG)

# packages
find_package(CUDA)
if(CUDA_FOUND)
    cuda_add_executable(cublas-bench "cublas.cu")
    target_link_libraries(cublas-bench "cublas")
endif()
