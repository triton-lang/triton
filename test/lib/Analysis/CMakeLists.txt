add_mlir_library(TritonTestAnalysis
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestMembar.cpp

  LINK_LIBS PUBLIC
  TritonAnalysis
)