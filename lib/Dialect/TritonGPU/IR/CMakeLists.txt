add_mlir_dialect_library(TritonGPUIR
  Dialect.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
