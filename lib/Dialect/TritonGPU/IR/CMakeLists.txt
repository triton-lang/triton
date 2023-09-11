add_mlir_dialect_library(TritonGPUIR
  Dialect.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  NVGPUTableGen
  NVGPUAttrDefsIncGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUOps
  TritonIR
)
