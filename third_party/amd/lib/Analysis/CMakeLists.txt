add_triton_library(TritonAMDAnalysis
  RangeAnalysis.cpp
  AxisInfoExt.cpp
  AMDGPUAllocation.cpp

  DEPENDS
  TritonTableGen
  TritonAMDGPUTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
