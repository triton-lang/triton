add_mlir_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  ASMBuilder
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
)
