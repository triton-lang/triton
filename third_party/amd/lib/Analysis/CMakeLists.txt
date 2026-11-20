add_triton_library(TritonAMDAnalysis
  RangeAnalysis.cpp
  AxisInfoExt.cpp
  AMDGPUAllocation.cpp
  Membar.cpp

  DEPENDS
  TritonTableGen
  TritonAMDGPUTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonAnalysis
  TritonIR
  TritonGPUIR
)
