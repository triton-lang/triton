#===------------------------------------------------------------------------===#
#
# Copyright (c) Triton Project Contributors.
#
#===------------------------------------------------------------------------===#

add_mlir_conversion_library(TritonToLinalg
  TritonToLinalg.cpp
  TritonToLinalgPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonToLinalg
  ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonToLinalg

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonAnalysis
  TritonIR
  TritonTransforms
)
