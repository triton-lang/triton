add_triton_library(TritonAMDAnalysis
  RangeAnalysis.cpp
  ControlFlowRangeAnalysis.cpp
  AxisInfoExt.cpp
  AMDGPUAllocation.cpp

  DEPENDS
  TritonTableGen
  TritonAMDGPUTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRControlFlowDialect
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
