cmake_minimum_required(VERSION 3.6)
include(ExternalProject)

set(TRITON_USE_ROCM "$ENV{TRITON_USE_ROCM}")

if(NOT TRITON_LLVM_BUILD_DIR)
    set(TRITON_LLVM_BUILD_DIR ${CMAKE_BINARY_DIR})
endif()


project(triton)
include(CTest)
if (TRITON_USE_ROCM)
    message("COMPILING WITH ROCM")
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${PROJECT_SOURCE_DIR}/third_party/hipify-torch/cmake")
    include(Hipify)
else()
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()

# Options
option(BUILD_TUTORIALS "Build C++ Triton tutorials" ON)
option(BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Default build type: Release")
  set(CMAKE_BUILD_TYPE "Release")
endif()

find_library(TERMINFO_LIBRARY tinfo)

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
if (TRITON_USE_ROCM)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -std=gnu++17 -Wno-unused-result -Wno-attributes")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -std=gnu++17")
endif()


##########
# LLVM
##########
if (TRITON_USE_ROCM)
    find_package(LLVM 12 REQUIRED COMPONENTS "amdgpu")
else()
    if("${LLVM_LIBRARY_DIR}" STREQUAL "")
        find_package(LLVM 11 REQUIRED COMPONENTS "nvptx")
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
        if(APPLE)
        set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14")
        endif()
    # sometimes we don't want to use llvm-config, since it may have been downloaded for some specific linux distros
    else()
        set(LLVM_LDFLAGS "-L${LLVM_LIBRARY_DIR}")
        set(LLVM_LIBRARIES libLLVMNVPTXCodeGen.a libLLVMSelectionDAG.a libLLVMipo.a libLLVMInstrumentation.a
                        libLLVMVectorize.a libLLVMLinker.a libLLVMIRReader.a libLLVMAsmParser.a libLLVMFrontendOpenMP.a
                        libLLVMAsmPrinter.a libLLVMDebugInfoDWARF.a libLLVMCodeGen.a libLLVMTarget.a libLLVMScalarOpts.a
                        libLLVMInstCombine.a libLLVMAggressiveInstCombine.a libLLVMTransformUtils.a libLLVMBitWriter.a
                        libLLVMAnalysis.a libLLVMProfileData.a libLLVMObject.a libLLVMTextAPI.a libLLVMMCParser.a
                        libLLVMBitReader.a libLLVMCore.a libLLVMRemarks.a libLLVMBitstreamReader.a libLLVMNVPTXDesc.a
                        libLLVMMC.a libLLVMDebugInfoCodeView.a libLLVMDebugInfoMSF.a libLLVMBinaryFormat.a libLLVMNVPTXInfo.a
                        libLLVMSupport.a libLLVMDemangle.a)
    endif()
endif()
include_directories("${LLVM_INCLUDE_DIRS}")

# Python module
if(BUILD_PYTHON_MODULE)
    message(STATUS "Adding Python module")
    # Build CUTLASS python wrapper if requested
    set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
    set(CUTLASS_INCLUDE_DIR "$ENV{CUTLASS_INCLUDE_DIR}")
    set(CUTLASS_LIBRARY_DIR "$ENV{CUTLASS_LIBRARY_DIR}")
    if(NOT("${CUTLASS_INCLUDE_DIR}" STREQUAL "") AND NOT("${CUTLASS_LIBRARY_DIR}" STREQUAL ""))
    if (TRITON_USE_ROCM)
        set(CUTLASS_SRC ${PYTHON_SRC_PATH}/cutlass_hip.cc)
    else()
        set(CUTLASS_SRC ${PYTHON_SRC_PATH}/cutlass.cc)
    endif()
        add_definitions(-DWITH_CUTLASS_BINDINGS)
        set(CUTLASS_LIBRARIES "cutlass.a")
    endif()
    message(STATUS ${CUTLASS_INCLUDE_PATH})
    include_directories("." ${PYTHON_SRC_PATH} ${PYTHON_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIR})
    link_directories(${PYTHON_LINK_DIRS} ${CUTLASS_LIBRARY_DIR})
    if (TRITON_USE_ROCM)
        add_definitions(-D__HIP_PLATFORM_AMD__)
        set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/triton_hip.cc  ${PYTHON_SRC_PATH}/superblock.cc ${CUTLASS_SRC})
    else()
        set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/triton.cc  ${PYTHON_SRC_PATH}/superblock.cc ${CUTLASS_SRC})
    endif()
endif()

# Triton
file(GLOB_RECURSE LIBTRITON_SRC lib/*.cc)

if (TRITON_USE_ROCM)
    # collect hip src files
    file(GLOB_RECURSE LIBTRITON_HIP_SRC *_hip.*)
    list(FILTER LIBTRITON_HIP_SRC EXCLUDE REGEX ".*triton/python/build.*$") # don't include file from build

    # get CUDA analog of HIP srcs
    set(LIBTRITON_CUDA_ANALOG_SRC ${LIBTRITON_CUDA_ANALOG_SRC})
    foreach(HIP_SRC IN LISTS LIBTRITON_HIP_SRC)
        string(REPLACE "_hip." "." CUDA_SRC ${HIP_SRC})
        list(APPEND LIBTRITON_CUDA_ANALOG_SRC ${CUDA_SRC})
    endforeach()

    # remove CUDA analogs of HIP srcs
    foreach(SRC IN LISTS LIBTRITON_SRC)
        if (${SRC} IN_LIST LIBTRITON_CUDA_ANALOG_SRC)
            list(REMOVE_ITEM LIBTRITON_SRC ${SRC})
        endif()
    endforeach()
else ()
    # fitler hip files
    file(GLOB_RECURSE LIBTRITON_HIP_SRC *_hip.*)
    foreach(SRC IN LISTS LIBTRITON_SRC)
        if (${SRC} IN_LIST LIBTRITON_HIP_SRC)
            list(REMOVE_ITEM LIBTRITON_SRC ${SRC})
        endif()
    endforeach()
endif()

add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
target_link_options(triton PRIVATE ${LLVM_LDFLAGS})
target_link_libraries(triton ${LLVM_LIBRARIES} z ${TERMINFO_LIBRARY})


if(BUILD_PYTHON_MODULE)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
    # Check if the platform is MacOS
    if(APPLE)
        set(PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
    endif()
    target_link_libraries(triton ${CUTLASS_LIBRARIES} ${PYTHON_LDFLAGS})
endif()