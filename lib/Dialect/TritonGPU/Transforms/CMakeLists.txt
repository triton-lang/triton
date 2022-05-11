add_mlir_dialect_library(TritonGPUTransforms
  Pipeline.cpp
  TritonGPUConversion.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
