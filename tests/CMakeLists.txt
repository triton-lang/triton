#helper
function(add_isaac_test NAME)
     set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${DIRECTORY}")
     add_executable(test-${NAME} ${NAME}.cpp)
     add_test(${NAME} "test-${NAME}")
     target_link_libraries(test-${NAME} isaac)
endfunction()
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

#tests
foreach(NAME gemm conv pool)
    add_isaac_test(${NAME})
endforeach()
