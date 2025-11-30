add_mlir_library(TritonTestAnalysis
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestBufferRegion.cpp
  TestMembar.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  TritonAnalysis
)
