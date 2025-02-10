add_triton_library(ProtonGPUIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ProtonGPUTableGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  ProtonIR
)
