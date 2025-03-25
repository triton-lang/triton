add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp
  UtilityXPU.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
)
