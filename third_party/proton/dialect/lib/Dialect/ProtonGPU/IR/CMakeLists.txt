add_triton_library(ProtonGPUIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  ProtonGPUTableGen
  ProtonGPUAttrDefsIncGen
  ProtonGPUTypesIncGen

  LINK_LIBS PUBLIC
  TritonGPUIR
  ProtonIR
)
