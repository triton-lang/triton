cmake_minimum_required(VERSION 3.6)
include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

if(NOT TRITON_LLVM_BUILD_DIR)
    set(TRITON_LLVM_BUILD_DIR ${CMAKE_BINARY_DIR})
endif()


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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third-party/pybind11/include)

if(WIN32)
    SET(BUILD_SHARED_LIBS OFF)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/deps/dlfcn-win32/src)
    add_subdirectory(deps/dlfcn-win32/src ${CMAKE_BINARY_DIR}/dlfcn-win32)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17 -fvisibility=hidden -fvisibility-inlines-hidden")
if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 11.6)
endif()



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
include_directories(${LLVM_INCLUDE_DIRS})

# Python module
if(BUILD_PYTHON_MODULE)
    message(STATUS "Adding Python module")
    set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
    include_directories("." ${PYTHON_SRC_PATH} ${PYTHON_INCLUDE_DIRS})
    link_directories(${PYTHON_LINK_DIRS})
    set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/triton.cc)
endif()


# # Triton
# file(GLOB_RECURSE LIBTRITON_SRC lib/*.cc)
# if (WIN32 AND BUILD_PYTHON_MODULE)
#     find_package(Python3 REQUIRED COMPONENTS Development)
#     Python3_add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
#     set_target_properties(triton PROPERTIES SUFFIX ".pyd")
#     set_target_properties(triton PROPERTIES PREFIX "lib")
# else()
#     add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
# endif()


# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${LLVM_LIBRARY_DIR}/cmake/mlir)

list(APPEND CMAKE_MODULE_PATH ${LLVM_LIBRARY_DIR}/cmake/llvm)
list(APPEND CMAKE_MODULE_PATH ${LLVM_LIBRARY_DIR}/cmake/mlir)

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)
# include(HandleLLVMOptions) # human-friendly error message

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files
# link_directories(${LLVM_LIBRARY_DIR})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

add_library(triton SHARED ${PYTHON_SRC})

# find_package(PythonLibs REQUIRED)

set(TRITON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

target_link_libraries(triton
  ${PYTHON_LIBRARIES}
  TritonAnalysis
  TritonTransforms
  TritonGPUTransforms
  TritonDriver
  TritonLLVMIR
  TritonPTX
  ${dialect_libs}
  ${conversion_libs}
  # optimizations
  MLIRPass
  MLIRTransforms
  MLIRLLVMIR
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRExecutionEngine
  MLIRMathToLLVM
  MLIRNVVMToLLVMIRTranslation
  MLIRIR
)

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

add_subdirectory(test)

add_subdirectory(unittest)
