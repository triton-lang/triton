include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()
include(AddMLIRPython)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=triton._C.triton_mlir_bindings.")

declare_mlir_python_sources(TritonPythonSources)
declare_mlir_python_sources(TritonPythonSources.Dialects ADD_TO_PARENT
                            TritonPythonSources)

declare_mlir_python_extension(
  TritonPythonSources.DialectExtension
  MODULE_NAME
  _tritonDialects
  ADD_TO_PARENT
  TritonPythonSources
  SOURCES
  TritonDialects.cpp
  EMBED_CAPI_LINK_LIBS
  TritonCAPI)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  TritonPythonSources.Dialects
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir_bindings"
  TD_FILE
  dialects/TritonOps.td
  SOURCES
  dialects/triton.py
  DIALECT_NAME
  tt)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT
  TritonPythonSources.Dialects
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir_bindings"
  TD_FILE
  dialects/TritonGPUOps.td
  SOURCES
  dialects/triton_gpu.py
  DIALECT_NAME
  triton_gpu)
declare_mlir_python_sources(
  TritonPythonSources.ExecutionEngine
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/triton_mlir_bindings"
  ADD_TO_PARENT
  TritonPythonSources
  SOURCES_GLOB
  triton/*.py)

# ##############################################################################
# Common CAPI
# ##############################################################################
add_mlir_python_common_capi_library(
  TritonPythonCAPI
  INSTALL_COMPONENT
  TritonPythonModules
  INSTALL_DESTINATION
  triton_mlir_bindings/_mlir_libs
  OUTPUT_DIRECTORY
  "${TRITON_BINARY_DIR}/python_packages/triton_mlir_bindings/_mlir_libs"
  RELATIVE_INSTALL_ROOT
  "../../../.."
  DECLARED_SOURCES
  TritonPythonSources
  MLIRPythonExtension.RegisterEverything
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine)

# ##############################################################################
# Instantiation of all Python modules
# ##############################################################################
add_mlir_python_modules(
  TritonPythonModules
  ROOT_PREFIX
  "${TRITON_BINARY_DIR}/python_packages/triton_mlir_bindings"
  INSTALL_PREFIX
  "triton_mlir_bindings"
  DECLARED_SOURCES
  TritonPythonSources
  LLVMSupport
  MLIRPythonExtension.RegisterEverything
  MLIRPythonSources.Core
  MLIRPythonSources.Dialects
  MLIRPythonSources.ExecutionEngine
  COMMON_CAPI_LINK_LIBS
  TritonPythonCAPI)

add_llvm_install_targets(install-triton-mlir-bindings DEPENDS
                         TritonPythonModules COMPONENT TritonPythonModules)
