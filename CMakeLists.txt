cmake_minimum_required(VERSION 3.20)

# Introduced in cmake 3.20
# https://cmake.org/cmake/help/latest/policy/CMP0116.html
cmake_policy(SET CMP0116 OLD)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

project(triton)
include(CTest)

if(NOT WIN32)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
endif()

# Options
option(TRITON_BUILD_TUTORIALS "Build C++ Triton tutorials" ON)
option(TRITON_BUILD_PYTHON_MODULE "Build Python Triton bindings" OFF)

# Ensure Python3 vars are set correctly
# used conditionally in this file and by lit tests

# Customized release build type with assertions: TritonRelBuildWithAsserts
set(CMAKE_C_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")
set(CMAKE_CXX_FLAGS_TRITONRELBUILDWITHASSERTS "-O2 -g")

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
  find_package(dlfcn-win32 REQUIRED)
  set(CMAKE_DL_LIBS dlfcn-win32::dl)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -D__STDC_FORMAT_MACROS  -fPIC -std=gnu++17 -fvisibility=hidden -fvisibility-inlines-hidden")

if(APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET 11.6)
endif()

# #########
# LLVM
# #########
if(NOT MLIR_DIR)
  if(NOT LLVM_LIBRARY_DIR)
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

    # FindLLVM outputs LLVM_LIBRARY_DIRS but we expect LLVM_LIBRARY_DIR here
    set(LLVM_LIBRARY_DIR ${LLVM_LIBRARY_DIRS})

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

  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()

# Python module
if(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Adding Python module")
  set(PYTHON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/src)
  set(PYTHON_SRC ${PYTHON_SRC_PATH}/main.cc ${PYTHON_SRC_PATH}/triton.cc)
  include_directories("." ${PYTHON_SRC_PATH})

  if(PYTHON_INCLUDE_DIRS)
    include_directories(${PYTHON_INCLUDE_DIRS})
  else()
    find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
    include_directories(${Python3_INCLUDE_DIRS})
    link_directories(${Python3_LIBRARY_DIRS})
    link_libraries(${Python3_LIBRARIES})
    add_link_options(${Python3_LINK_OPTIONS})
  endif()
endif()

# # Triton
# file(GLOB_RECURSE LIBTRITON_SRC lib/*.cc)
# if (WIN32 AND TRITON_BUILD_PYTHON_MODULE)
# Python3_add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
# set_target_properties(triton PROPERTIES SUFFIX ".pyd")
# set_target_properties(triton PROPERTIES PREFIX "lib")
# else()
# add_library(triton SHARED ${LIBTRITON_SRC} ${PYTHON_SRC})
# endif()

# MLIR
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen) # required by AddMLIR
include(AddLLVM)
include(AddMLIR)

# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wno-covered-switch-default")

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files

# link_directories(${LLVM_LIBRARY_DIR})
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

# find_package(PythonLibs REQUIRED)
set(TRITON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TRITON_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(TRITON_BUILD_PYTHON_MODULE)
  add_library(triton SHARED ${PYTHON_SRC})
  set(TRITON_LIBRARIES
    TritonAnalysis
    TritonTransforms
    TritonGPUTransforms
    TritonLLVMIR
    TritonPTX
    ${dialect_libs}
    ${conversion_libs}

    # optimizations
    MLIRPass
    MLIRTransforms
    MLIRLLVMDialect
    MLIRSupport
    MLIRTargetLLVMIRExport
    MLIRExecutionEngine
    MLIRMathToLLVM
    MLIRNVVMToLLVMIRTranslation
    MLIRIR
  )

  if(WIN32)
    target_link_libraries(triton PRIVATE ${LLVM_LIBRARIES} ${CMAKE_DL_LIBS}
      ${TRITON_LIBRARIES}
    )
  elseif(APPLE)
    target_link_libraries(triton ${LLVM_LIBRARIES} z
      ${TRITON_LIBRARIES}
    )
  else()
    target_link_libraries(triton ${LLVM_LIBRARIES} z stdc++fs
      ${TRITON_LIBRARIES}
    )
  endif()

  target_link_options(triton PRIVATE ${LLVM_LDFLAGS})
endif()

if(UNIX AND NOT APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--exclude-libs,ALL")
endif()

if(TRITON_BUILD_PYTHON_MODULE AND NOT WIN32)
  set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")

  # Check if the platform is MacOS
  if(APPLE)
    set(PYTHON_LDFLAGS "-undefined dynamic_lookup -flto")
  endif()

  target_link_libraries(triton ${CUTLASS_LIBRARIES} ${PYTHON_LDFLAGS})
endif()

add_subdirectory(test)

add_subdirectory(unittest)
