add_triton_library(ProtonIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ProtonGPUTableGen
  ProtonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  ProtonGPUIR
)
