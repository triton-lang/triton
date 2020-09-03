cmake_minimum_required(VERSION 2.8)
project(triton)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_TESTS "Build C++ Triton tests" ON)
option(BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)

# LLVM
find_package(LLVM REQUIRED)
link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS -std=gnu++11")

# Tests
if(BUILD_TESTS)
  message(STATUS "Adding C++ tests")
  add_subdirectory(tests)
endif() 

# Python module
if(BUILD_PYTHON_MODULE)
    message(STATUS "Adding Python module")
    # PyBind11 wrapper source file
    set(PYTHON_SRC bindings.cc launch.cc)
    set_source_files_properties(launch.cc PROPERTIES COMPILE_FLAGS "-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=0")
    include_directories("." ${PYTHON_INCLUDE_DIRS})
    link_directories(${PYTHON_LINK_DIRS})
endif()


# Triton
file(GLOB_RECURSE LIBTRITON_SRC lib/*.cc)
add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
target_link_libraries(triton ${LLVM_LIBRARIES})

if(BUILD_PYTHON_MODULE)
    target_link_libraries(triton ${TORCH_LIBRARIES})
endif()
