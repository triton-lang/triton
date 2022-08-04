add_mlir_library(TritonTestAnalysis
  TestAxisInfo.cpp
  TestAllocation.cpp

  LINK_LIBS PUBLIC
  TritonAnalysis
)