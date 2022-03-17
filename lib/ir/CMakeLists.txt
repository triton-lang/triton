add_mlir_dialect_library(TRITONIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithmetic
  MLIRControlFlow
  MLIRFunc
  MLIRTensor
)
