cmake_minimum_required(VERSION 2.8)

# Add visibility of headers
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake_modules")

find_package(ViennaCL QUIET REQUIRED)
include_directories(${VIENNACL_INCLUDE_DIRS})

find_package(OpenCL QUIET REQUIRED)

include_directories(${OPENCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR})

INCLUDE(CTest)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(python)
