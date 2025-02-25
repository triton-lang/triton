add_mlir_library(TritonAMDGPUTestAnalysis
  TestAMDRangeAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
