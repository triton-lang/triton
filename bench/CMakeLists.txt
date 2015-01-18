set(CMAKE_BUILD_TYPE Release)

find_package(CUDA)

foreach(PROG blas overhead)
     add_executable(${PROG}-bench  ${PROG}.cpp)
     target_link_libraries(${PROG}-bench atidlas ${OPENCL_LIBRARIES})
     set_target_properties(${PROG}-bench PROPERTIES COMPILE_FLAGS "-Wall -Wextra")

     if(CUDA_FOUND)
         set(CUPROG cu${PROG})
         cuda_add_executable(${CUPROG}-bench ${CUPROG}.cu)
         target_link_libraries(${CUPROG}-bench ${CUPROG})
     endif()
endforeach(PROG)
