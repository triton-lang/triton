add_mlir_dialect_library(TritonIR
  Interfaces.cpp
  Dialect.cpp
  Ops.cpp
  Types.cpp
  Traits.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithmetic
  MLIRSCF

  # Since LLVM 15
  # MLIRFunc
  # else
  MLIRStandard
)
