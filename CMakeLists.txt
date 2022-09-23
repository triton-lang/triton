cmake_minimum_required(VERSION 3.6)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

project(triton)
include(CTest)
if(NOT WIN32)
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

if(NOT WIN32)
    find_library(TERMINFO_LIBRARY tinfo)
endif()

# Compiler flags
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Third-party
include_directories(${PYBIND11_INCLUDE_DIR})

if(WIN32)
    SET(BUILD_SHARED_LIBS OFF)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/dlfcn-win32/src)
    add_subdirectory(deps/dlfcn-win32/src ${CMAKE_BINARY_DIR}/dlfcn-win32)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_FORMAT_MACROS  -std=gnu++17")


##########
# LLVM
##########
if("${LLVM_LIBRARY_DIR}" STREQUAL "")
    if(WIN32)
      find_package(LLVM 13 REQUIRED COMPONENTS nvptx amdgpu)

      include_directories(${LLVM_INCLUDE_DIRS})
      separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
      add_definitions(${LLVM_DEFINITIONS_LIST})

      llvm_map_components_to_libnames(LLVM_LIBRARIES support core
        NVPTXInfo nvptxcodegen
        AMDGPUInfo AMDGPUcodegen
      )
    else()
      find_package(LLVM 11 REQUIRED COMPONENTS "nvptx;amdgpu")
    endif()
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    if(APPLE)
      set(CMAKE_OSX_DEPLOYMENT_TARGET "10.14")
    endif()
# sometimes we don't want to use llvm-config, since it may have been downloaded for some specific linux distros
else()
    set(LLVM_LDFLAGS "-L${LLVM_LIBRARY_DIR}")
    set(LLVM_LIBRARIES
libLLVMNVPTXCodeGen.a
libLLVMNVPTXDesc.a
libLLVMNVPTXInfo.a
libLLVMAMDGPUDisassembler.a
libLLVMMCDisassembler.a
libLLVMAMDGPUCodeGen.a
libLLVMMIRParser.a
libLLVMGlobalISel.a
libLLVMSelectionDAG.a
libLLVMipo.a
libLLVMInstrumentation.a
libLLVMVectorize.a
libLLVMLinker.a
libLLVMIRReader.a
libLLVMAsmParser.a
libLLVMFrontendOpenMP.a
libLLVMAsmPrinter.a
libLLVMDebugInfoDWARF.a
libLLVMCodeGen.a
libLLVMTarget.a
libLLVMScalarOpts.a
libLLVMInstCombine.a
libLLVMAggressiveInstCombine.a
libLLVMTransformUtils.a
libLLVMBitWriter.a
libLLVMAnalysis.a
libLLVMProfileData.a
libLLVMObject.a
libLLVMTextAPI.a
libLLVMBitReader.a
libLLVMAMDGPUAsmParser.a
libLLVMMCParser.a
libLLVMAMDGPUDesc.a
libLLVMAMDGPUUtils.a
libLLVMMC.a
libLLVMDebugInfoCodeView.a
libLLVMDebugInfoMSF.a
libLLVMCore.a
libLLVMRemarks.a
libLLVMBitstreamReader.a
libLLVMBinaryFormat.a
libLLVMAMDGPUInfo.a
libLLVMSupport.a
libLLVMDemangle.a
libLLVMPasses.a
libLLVMAnalysis.a
libLLVMTransformUtils.a
libLLVMScalarOpts.a
libLLVMTransformUtils.a
libLLVMipo.a
libLLVMObjCARCOpts.a
libLLVMCoroutines.a
libLLVMAnalysis.a
)
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
        set(CUTLASS_SRC ${PYTHON_SRC_PATH}/cutlass.cc)
        add_definitions(-DWITH_CUTLASS_BINDINGS)
        set(CUTLASS_LIBRARIES "cutlass.a")
    endif()
    include_directories("." ${PYTHON_SRC_PATH} ${PYTHON_INCLUDE_DIRS} ${CUTLASS_INCLUDE_DIR})
    link_directories(${PYTHON_LINK_DIRS} ${CUTLASS_LIBRARY_DIR})
    set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/triton.cc  ${PYTHON_SRC_PATH}/superblock.cc ${CUTLASS_SRC})
endif()


# Triton
file(GLOB_RECURSE LIBTRITON_SRC lib/*.cc)
if (WIN32 AND BUILD_PYTHON_MODULE)
    find_package(Python3 REQUIRED COMPONENTS Development)
    Python3_add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
    set_target_properties(triton PROPERTIES SUFFIX ".pyd")
    set_target_properties(triton PROPERTIES PREFIX "lib")
else()
    add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
endif()

target_link_options(triton PRIVATE ${LLVM_LDFLAGS})

if(WIN32)
    target_link_libraries(triton PRIVATE ${LLVM_LIBRARIES} dl) # dl is from dlfcn-win32
else()
    target_link_libraries(triton ${LLVM_LIBRARIES} z)
endif()


if(BUILD_PYTHON_MODULE AND NOT WIN32)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
    # Check if the platform is MacOS
    if(APPLE)
        set(PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
    endif()
    target_link_libraries(triton ${CUTLASS_LIBRARIES} ${PYTHON_LDFLAGS})
endif()
