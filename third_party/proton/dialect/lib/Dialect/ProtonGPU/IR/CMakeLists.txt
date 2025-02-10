add_triton_library(ProtonGPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ProtonGPUTableGen
  ProtonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  ProtonIR
)
