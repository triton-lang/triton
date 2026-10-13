include(CheckSymbolExists)
# Find OneDNN ukernel library
find_package(dnnl CONFIG)
if (dnnl_FOUND)
  get_target_property(dnnl_include DNNL::dnnl INTERFACE_INCLUDE_DIRECTORIES)
  # currently used only in triton_cpu.cc and in ConvertDotToOneDNN
  include_directories(${dnnl_include})
  set (dnnl_header ${dnnl_include}/oneapi/dnnl/dnnl_config.h)
  check_symbol_exists(DNNL_EXPERIMENTAL_UKERNEL ${dnnl_header}  HAS_UKERNELS)
  if (HAS_UKERNELS)
    message(STATUS "Found OneDNN/DNNL with Ukernels support")
    add_compile_definitions(ONEDNN_AVAILABLE)
  else ()
    message(STATUS "SKIPPING found OneDNN/DNNL because Ukernels support is missing")
  endif()
else ()
  message(STATUS "Could NOT find OneDNN/DNNL")
endif()

# Find XSMM ukernel library
find_library(LIBXSMM xsmm
  PATHS ENV XSMM_LIBRARY_DIRS
)
find_path(LIBXSMM_INCLUDE_DIR
  NAMES libxsmm.h
  PATHS ENV XSMM_INCLUDE_DIRS
)
if (LIBXSMM AND LIBXSMM_INCLUDE_DIR)
  set(xsmm_FOUND True)
  message(STATUS "Found XSMM: " ${LIBXSMM})
  add_compile_definitions(XSMM_AVAILABLE)
  include_directories(${LIBXSMM_INCLUDE_DIR})
else ()
  message(STATUS "Could NOT find XSMM")
endif()

# Build Triton-CPU plugin
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  if(NOT "${LLVM_NATIVE_ARCH}" IN_LIST LLVM_TARGETS_TO_BUILD)
    message(FATAL_ERROR "TritonCPU requires the LLVM native target (${LLVM_NATIVE_ARCH})")
  endif()
  llvm_map_components_to_libnames(TRITON_CPU_NATIVE_LLVM_LIBS nativecodegen)

  add_triton_plugin(TritonCPU
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_cpu.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/llvm.cc
    LINK_LIBS TritonToTritonCPU TritonCPUToLLVM TritonCPUTransforms
  )
  target_link_libraries(TritonCPU
    PUBLIC
      MLIRVectorToSCF
      MLIRAffineToStandard
      MLIRArmNeonToLLVMIRTranslation
      MLIRMathDialect
      MLIRMathToLLVM
      MLIRMathToLibm
      MLIRMemRefTransforms
      MLIRReconcileUnrealizedCasts
      ${TRITON_CPU_NATIVE_LLVM_LIBS}
      LLVMCore
      LLVMipo
      LLVMIRReader
      LLVMMC
      LLVMSupport
      LLVMTarget
      LLVMTargetParser
    PRIVATE
      Python3::Module
      nanobind-static
  )
endif()

# Configure and build Triton-CPU runtime
set(TRITON_CPU_RUNTIME_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/runtime/cpu_runtime.cpp)
set(TRITON_CPU_RUNTIME_LIBS LLVMSupport)
if (dnnl_FOUND AND HAS_UKERNELS)
  set(TRITON_CPU_RUNTIME_SOURCES ${TRITON_CPU_RUNTIME_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_onednn.cpp)
  set(TRITON_CPU_RUNTIME_LIBS ${TRITON_CPU_RUNTIME_LIBS} DNNL::dnnl)
endif()
if(xsmm_FOUND)
  set(TRITON_CPU_RUNTIME_SOURCES ${TRITON_CPU_RUNTIME_SOURCES} ${CMAKE_CURRENT_SOURCE_DIR}/runtime/runtime_xsmm.cpp)
  set(TRITON_CPU_RUNTIME_LIBS ${TRITON_CPU_RUNTIME_LIBS} ${LIBXSMM})
endif()

add_library(TritonCPURuntime SHARED ${TRITON_CPU_RUNTIME_SOURCES})
target_link_libraries(TritonCPURuntime PRIVATE ${TRITON_CPU_RUNTIME_LIBS})

# Build and link sleef
include(FetchContent)

set(TRITON_SLEEF_SOURCE_DIR "" CACHE PATH "Use a local SLEEF source checkout")
if(TRITON_SLEEF_SOURCE_DIR)
  set(FETCHCONTENT_SOURCE_DIR_SLEEF "${TRITON_SLEEF_SOURCE_DIR}")
elseif(TRITON_OFFLINE_BUILD)
  message(FATAL_ERROR
    "The CPU backend requires SLEEF. Set TRITON_SLEEF_SOURCE_DIR when "
    "TRITON_OFFLINE_BUILD is enabled.")
endif()

set(SLEEF_BUILD_SHARED_LIBS ON CACHE BOOL "Build sleef shared lib" FORCE)
set(SLEEF_BUILD_DFT OFF CACHE BOOL "Don't build sleef DFT lib" FORCE)
set(SLEEF_BUILD_GNUABI_LIBS OFF CACHE BOOL "Don't build sleef gnuabi libs" FORCE)
set(SLEEF_BUILD_TESTS OFF CACHE BOOL "Don't build sleef tests" FORCE)
set(SLEEF_BUILD_SCALAR_LIB OFF CACHE BOOL "libsleefscalar will not be built." FORCE)
FetchContent_Declare(
  sleef
  GIT_REPOSITORY https://github.com/shibatch/sleef.git
  GIT_TAG 93f04d869471ce4d007abaebb8c6a7bc62749f61
)
FetchContent_MakeAvailable(sleef)
# Override sleef's output directory with our own
set_target_properties(sleef PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
