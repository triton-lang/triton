add_triton_library(TritonNvidiaGPUIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonNvidiaGPUTableGen
  TritonNvidiaGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
