add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  BufferRegion.cpp
  BufferIndexAnalysis.cpp
  Membar.cpp
  MembarCFG.cpp
  MembarCut.cpp
  MembarDependencies.cpp
  MembarPlacement.cpp
  MembarSMem.cpp
  MembarTMem.cpp
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
  TritonInstrumentIR
  TritonNvidiaGPUIR
)
