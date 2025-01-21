add_triton_library(TritonNvidiaGPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TritonNvidiaGPUTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
