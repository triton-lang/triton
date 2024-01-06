add_mlir_dialect_library(TritonIR
  Dialect.cpp
  Ops.cpp
  Types.cpp
  Traits.cpp

  DEPENDS
  TritonTableGen
)
