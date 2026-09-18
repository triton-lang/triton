add_triton_library(TritonAMDAnalysis
  AxisInfoExt.cpp
  AMDGPUAllocation.cpp

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
