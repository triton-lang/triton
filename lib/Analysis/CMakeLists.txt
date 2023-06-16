add_mlir_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp
  UseAnalysis.cpp
  MaskAnalysis.cpp
  PtrAnalysis.cpp
  OpFoldResultUtils.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
)
