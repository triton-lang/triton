add_triton_library(NVWSIR
  Dialect.cpp
  Ops.cpp
  SemaphorePendingCount.cpp

  DEPENDS
  NVWSTableGen
  NVWSAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
