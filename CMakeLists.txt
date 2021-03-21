cmake_minimum_required(VERSION 2.8)
include(ExternalProject)


project(triton)
include(CTest)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Options
option(BUILD_TUTORIALS "Build C++ Triton tutorials" ON)
option(BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS -fvisibility=default -std=gnu++14")



##########
# LLVM
##########
get_cmake_property(_variableNames VARIABLES)
set(__variableNames ${_variableNames})

configure_file(cmake/DownloadLLVM.in llvm-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/llvm-download"
)
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/llvm-download"
)
set(LLVM_TARGETS_TO_BUILD "NVPTX" CACHE INTERNAL "")
set(LLVM_BUILD_RUNTIME "OFF" CACHE INTERNAL "")
set(LLVM_BUILD_RUNTIMES "OFF" CACHE INTERNAL "")
set(LLVM_BUILD_TOOLS "OFF" CACHE INTERNAL "")
set(LLVM_BUILD_UTILS "OFF" CACHE INTERNAL "")
set(LLVM_INCLUDE_BENCHMARKS "OFF" CACHE INTERNAL "")
set(LLVM_INCLUDE_DOCS "OFF" CACHE INTERNAL "")
set(LLVM_INCLUDE_EXAMPLES "OFF" CACHE INTERNAL "")
set(LLVM_INCLUDE_GO_TESTS "OFF" CACHE INTERNAL "")
set(LLVM_INCLUDE_RUNTIME "OFF" CACHE INTERNAL "")
set(LLVM_INCLUDE_TESTS "OFF" CACHE INTERNAL "")
set(LLVM_INCLUDE_TOOLS "OFF" CACHE INTERNAL "")
set(LLVM_INCLUDE_UTILS "OFF" CACHE INTERNAL "")
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/llvm-src
                 ${CMAKE_CURRENT_BINARY_DIR}/llvm-build)
get_property(LLVM_LIBRARIES GLOBAL PROPERTY LLVM_COMPONENT_LIBS)
# remove LLVM-specific variables so we don't pollute GUI
get_cmake_property(_variableNames VARIABLES)
list(REMOVE_ITEM _variableNames ${__variableNames})
list(REMOVE_ITEM _variableNames ${LLVM_LIBRARIES})
foreach (_variableName ${_variableNames})
    unset(${_variableName} CACHE)
endforeach()
include_directories("${CMAKE_CURRENT_BINARY_DIR}/llvm-build/include/"
                    "${CMAKE_CURRENT_BINARY_DIR}/llvm-src/include/")

# Python module
if(BUILD_PYTHON_MODULE)
    message(STATUS "Adding Python module")
    # Build CUTLASS python wrapper if requested
    set(CUTLASS_INCLUDE_DIR "$ENV{CUTLASS_INCLUDE_DIR}")
    set(CUTLASS_LIBRARY_DIR "$ENV{CUTLASS_LIBRARY_DIR}")
    if(NOT("${CUTLASS_INCLUDE_DIR}" STREQUAL "") AND NOT("${CUTLASS_LIBRARY_DIR}" STREQUAL ""))
        set(CUTLASS_SRC cutlass.cc)
        add_definitions(-DWITH_CUTLASS_BINDINGS)
        set(CUTLASS_LIBRARIES "cutlass.a")
    endif()
    message(STATUS ${CUTLASS_INCLUDE_PATH})
    include_directories("." ${PYTHON_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIR})
    link_directories(${PYTHON_LINK_DIRS} ${CUTLASS_LIBRARY_DIR})
    set(PYTHON_SRC main.cc triton.cc superblock.cc ${CUTLASS_SRC})
endif()


# Triton
file(GLOB_RECURSE LIBTRITON_SRC lib/*.cc)
add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
add_dependencies(triton llvm)
target_link_libraries(triton ${LLVM_LIBRARIES})

if(BUILD_PYTHON_MODULE)
    target_link_libraries(triton ${TORCH_LIBRARIES} ${CUTLASS_LIBRARIES})
endif()

# Tutorials
if(BUILD_TUTORIALS)
  message(STATUS "Adding C++ tutorials")
  add_subdirectory(tutorials)
endif()
