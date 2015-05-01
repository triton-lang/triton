cmake_minimum_required(VERSION 2.8.10)

# Add visibility of headers
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(OpenCL QUIET REQUIRED)
if(CUDA_FOUND)
   set(BACKEND_DEFINES "-DISAAC_WITH_CUDA")
   include_directories(${CUDA_INCLUDE_DIRS})
endif()


include(CTest)
add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(bench)
find_program(PYTHON "python")
if(PYTHON)
    add_subdirectory(python)
    add_subdirectory(tune)
endif()
