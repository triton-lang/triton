add_mlir_dialect_library(TritonGPUIR
  Dialect.cpp
  Traits.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
