add_triton_library(TritonCPUIR
  Dialect.cpp
  Ops.cpp
  Types.cpp

  DEPENDS
  TritonCPUTableGen
  TritonCPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
)
