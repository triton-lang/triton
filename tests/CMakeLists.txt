#cross-compiling for Android
if(ANDROID)
    add_subdirectory(android)
#not cross-compiling
else()

    #helper
    function(add_isaac_test DIRECTORY NAME)
         set(OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${DIRECTORY}")
         add_executable(test-${NAME} ${DIRECTORY}/${NAME}.cpp)
         add_test(${NAME} "test-${NAME}")
         target_link_libraries(test-${NAME} isaac)
    endfunction()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

    #api/blas
    foreach(NAME blas-1 blas-2 blas-3)
        add_isaac_test("api/blas" ${NAME})
    endforeach()
    #api/cpp
    foreach(NAME element-1d element-2d reduce-1d reduce-2d)
        add_isaac_test("api/cpp" ${NAME})
    endforeach()
    #runtime
    foreach(NAME fusion)
        add_isaac_test("runtime" ${NAME})
    endforeach()
endif()
