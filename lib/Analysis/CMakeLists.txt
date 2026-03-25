add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  BufferRegion.cpp
  Membar.cpp
  Alias.cpp
  RangeAnalysis.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithTransforms
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  GluonIR
  TritonNvidiaGPUIR
)
