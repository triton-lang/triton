add_mlir_dialect_library(TritonGPUConversion
  TritonGPUConversion.cpp

  # ADDITIONAL_HEADER_DIRS

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  # MLIRTransformUtils
)
