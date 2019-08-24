cmake_minimum_required(VERSION 2.8)
project(triton)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_EXAMPLES "Build C++ Triton examples" ON)
option(BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)

# LLVM
find_package(LLVM REQUIRED CONFIG)
include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
#llvm_map_components_to_libnames(llvm_libs all)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Gather headers for cmake-based IDEs
file( GLOB_RECURSE ALL_SRC *.cpp *.hpp *.h *.py CMakeLists*)
add_custom_target( ALL SOURCES ${ALL_SRC} )

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXXFLAGS} -std=c++11")

# Examples
if(BUILD_EXAMPLES)
  message(STATUS "Adding C++ examples")
  add_subdirectory(examples)
endif() 

# Python module
if(BUILD_PYTHON_MODULE)
    message(STATUS "Adding Python module")
    # PyBind11 wrapper source file
    file(GLOB_RECURSE PYTHON_SRC python/src/tensorflow.cpp)
    # update include directory
    include_directories(python/src/ ${PYTHON_INCLUDE_DIRS} ${TF_INCLUDE_DIRS})
    # update link directories
    link_directories(${TF_LIB_DIRS})
    # extra tensorflow ops (e.g., alloc_empty)
    file(GLOB_RECURSE EXTRA_TF_OPS_SRC python/src/tensorflow/*.cpp)
    add_library(extra_tf_ops SHARED ${EXTRA_TF_OPS_SRC})
    target_link_libraries(extra_tf_ops ${TF_LIBS})
endif()


# Triton
file(GLOB_RECURSE LIBTRITON_SRC lib/*.cpp lib/*.cc)
add_library(triton SHARED ${LIBTRITON_SRC} ${EIGHTCC_SRC} ${PYTHON_SRC})
target_link_libraries(triton LLVM)

# Warning level
#if(MSVC)
#  target_compile_options(triton PRIVATE /W4)
#else()
#  target_compile_options(triton PRIVATE -Wno-unused-parameter -Wall -Wextra -pedantic)
#endif()


