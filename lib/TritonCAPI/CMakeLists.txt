include_directories(${TRITON_BINARY_DIR}/include)
include_directories(${TRITON_SOURCE_DIR}/include)

add_mlir_public_c_api_library(
  TritonCAPI
  Dialects.cpp
  LINK_LIBS
  PUBLIC
  MLIRCAPIIR
  MLIRPass
  TritonGPUIR
  TritonGPUToLLVM
  TritonGPUTransforms
  TritonIR
  TritonToTritonGPU
  TritonTransforms)
