add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  BufferRegion.cpp
  BufferIndexAnalysis.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  TritonIR
  TritonGPUIR
  GluonIR
  TritonNvidiaGPUIR
)
