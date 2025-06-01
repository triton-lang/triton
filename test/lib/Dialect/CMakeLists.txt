add_mlir_library(TritonTestDialect
  TestLoopPeeling.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  TritonTransforms
)
