add_triton_library(TritonAMDAnalysis
  RangeAnalysis.cpp
  AxisInfoExt.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
