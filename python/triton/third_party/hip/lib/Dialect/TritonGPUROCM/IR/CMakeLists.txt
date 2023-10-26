add_mlir_dialect_library(TritonGPUROCMIR
  Dialect.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  TritonGPUROCMTableGen
  TritonGPUROCMAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUOps
  TritonIR
)
