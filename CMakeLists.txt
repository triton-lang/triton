cmake_minimum_required(VERSION 2.8)
project(triton)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_TUTORIALS "Build C++ Triton tutorials" ON)
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS -fvisibility=default -std=gnu++14")

# Tests
if(BUILD_TUTORIALS)
  message(STATUS "Adding C++ tutorials")
  add_subdirectory(tutorials)
endif() 

# Python module
if(BUILD_PYTHON_MODULE)
    message(STATUS "Adding Python module")
    # PyBind11 wrapper source file
    file(GLOB_RECURSE TORCH_SRC torch/*.cc)
    # Build CUTLASS python wrapper if requested
    set(CUTLASS_INCLUDE_DIR "$ENV{CUTLASS_INCLUDE_DIR}")
    set(CUTLASS_LIBRARY_DIR "$ENV{CUTLASS_LIBRARY_DIR}")
    if(NOT("${CUTLASS_INCLUDE_DIR}" STREQUAL "") AND NOT("${CUTLASS_LIBRARY_DIR}" STREQUAL ""))
        set(TORCH_SRC ${TORCH_SRC} cutlass.cc)
        add_definitions(-DWITH_CUTLASS_BINDINGS)
        set(CUTLASS_LIBRARIES "cutlass")
    endif()
    message(STATUS ${CUTLASS_INCLUDE_PATH})
    set(PYTHON_SRC main.cc triton.cc ${TORCH_SRC})
    set_source_files_properties(${TORCH_SRC} PROPERTIES COMPILE_FLAGS "-std=c++14 -D_GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI} ${CUTLASS_OPT}")
    include_directories("." ${PYTHON_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIR})
    link_directories(${PYTHON_LINK_DIRS} ${CUTLASS_LIBRARY_DIR})
endif()


# Triton
file(GLOB_RECURSE LIBTRITON_SRC lib/*.cc)
add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
target_link_libraries(triton ${LLVM_LIBRARIES} ${LLVM_SYSTEM_LIBS})

if(BUILD_PYTHON_MODULE)
    target_link_libraries(triton ${TORCH_LIBRARIES} ${CUTLASS_LIBRARIES})
endif()
