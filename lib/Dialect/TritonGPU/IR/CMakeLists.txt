add_mlir_dialect_library(TritonGPUIR
  Dialect.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
)
