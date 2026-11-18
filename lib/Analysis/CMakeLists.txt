add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  BufferRegion.cpp
  BufferIndexAnalysis.cpp
  Membar.cpp
  Membar/AsyncFenceModel.cpp
  Membar/CFGBuilder.cpp
  Membar/ClusterBarrierModel.cpp
  Membar/Solver.cpp
  Membar/Dependencies.cpp
  Membar/Pass.cpp
  Membar/SmemModel.cpp
  Membar/TmemModel.cpp
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
