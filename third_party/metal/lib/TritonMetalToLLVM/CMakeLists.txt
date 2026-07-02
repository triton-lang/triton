add_triton_library(TritonMetalToLLVM
  TargetInfo.cpp

  LINK_LIBS
  TritonAnalysis
  TritonIR
  TritonGPUIR
  TritonGPUToLLVM
  MLIRLLVMDialect
  MLIRIR
)
