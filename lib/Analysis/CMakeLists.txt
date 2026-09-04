add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  BufferRegion.cpp
  BufferIndexAnalysis.cpp
  Membar.cpp
  Alias.cpp
  RangeAnalysis.cpp
  ControlFlowRangeAnalysis.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRControlFlowDialect
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  GluonIR
  TritonNvidiaGPUIR
)
