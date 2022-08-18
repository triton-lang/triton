add_mlir_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp

  DEPENDS
  TritonGPUAttrDefsIncGen
)