set(CMAKE_BUILD_TYPE Release)

find_package(CUDA)

foreach(PROG blas)
     add_executable(${PROG}-bench ${PROG}.cpp)
     target_link_libraries(${PROG}-bench ${OPENCL_LIBRARIES})
     set_target_properties(${PROG}-bench PROPERTIES COMPILE_FLAGS "-DVIENNACL_WITH_OPENCL -Wall -Wextra")

     if(CUDA_FOUND)
         set(CUPROG cu${PROG})
         cuda_add_executable(${CUPROG}-bench ${CUPROG}.cu OPTIONS "-DVIENNACL_WITH_OPENCL")
         target_link_libraries(${CUPROG}-bench ${CUPROG} OpenCL)
     endif()
endforeach(PROG)
