add_triton_library(TritonAMDAnalysis
  RangeAnalysis.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)

add_triton_library(TritonAMDAxisInfoExt
  AxisInfoExt.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  TritonAMDGPUIR
)
