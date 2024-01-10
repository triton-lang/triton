add_triton_library(TritonIR
  Dialect.cpp
  Ops.cpp
  Types.cpp
  Traits.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
)
