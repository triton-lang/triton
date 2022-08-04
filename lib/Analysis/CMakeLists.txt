add_mlir_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp

  DEPENDS
  TritonGPUAttrDefsIncGen
)