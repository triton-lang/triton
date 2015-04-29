set(CMAKE_BUILD_TYPE RELEASE)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

file( GLOB_RECURSE LIBISAAC_SRC *.cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(isaac SHARED ${LIBISAAC_SRC})
set_target_properties(isaac PROPERTIES COMPILE_FLAGS "-Wno-sign-compare -D__CL_ENABLE_EXCEPTIONS ${BACKEND_DEFINES} -Wall -Wextra -pedantic -std=c++11")

if(OPENCL_FOUND)
    target_link_libraries(isaac ${OPENCL_LIBRARIES})
endif()
if(CUDA_FOUND)
    cuda_find_library_local_first(CUDA_NVRTC_LIBRARY nvrtc "\"nvrtc\" library")
    target_link_libraries(isaac ${CUDA_CUDA_LIBRARY} ${CUDA_NVRTC_LIBRARY})
endif()

install(TARGETS isaac LIBRARY DESTINATION lib)
set(INSTALL_INCLUDE_DIR /usr/local/include)
install(DIRECTORY isaac "${PROJECT_SOURCE_DIR}/include/isaac"
        DESTINATION "${INSTALL_INCLUDE_DIR}"  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")

set(POSTINSTALL_PY "${CMAKE_CURRENT_BINARY_DIR}/postinstall.py")
configure_file("${PROJECT_SOURCE_DIR}/python/postinstall.py" ${POSTINSTALL_PY})
install(CODE "execute_process(COMMAND ${PYTHON} ${POSTINSTALL_PY})")
