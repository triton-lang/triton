function(add_isaac_test NAME)
     add_executable(test-${NAME} ${NAME}.cpp)
     #set_target_properties(test-${NAME} PROPERTIES OUTPUT_NAME ${NAME})
     add_test(test-${NAME} test-${NAME})
     target_link_libraries(test-${NAME} isaac)
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(operations)
