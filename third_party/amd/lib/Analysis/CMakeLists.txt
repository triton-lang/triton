add_triton_library(TritonAMDAnalysis
  RangeAnalysis.cpp
  AxisInfoExt.cpp
  AMDGPUAllocation.cpp

  DEPENDS
  TritonTableGen
  TritonAMDGPUTableGen

)
