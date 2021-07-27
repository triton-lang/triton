cmake_minimum_required(VERSION 2.8)
project(triton)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_TESTS "Build C++ Triton tests" ON)
option(BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)

# LLVM
find_package(LLVM REQUIRED)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Tests
if(BUILD_TESTS)
  message(STATUS "Adding C++ tests")
  add_subdirectory(tests)
endif() 

# Python module
if(BUILD_PYTHON_MODULE)
    message(STATUS "Adding Python module")
    # PyBind11 wrapper source file
    file(GLOB_RECURSE PYTHON_SRC python/src/bindings.cc)
    include_directories(python/src/ ${PYTHON_INCLUDE_DIRS})
endif()


# Triton
file(GLOB_RECURSE LIBTRITON_SRC lib/*.cc)
add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
link_directories(${LLVM_LIBRARY_DIRS})
target_link_libraries(triton ${LLVM_LIBRARIES})

