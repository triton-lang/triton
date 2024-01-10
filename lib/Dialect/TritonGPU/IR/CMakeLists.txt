add_triton_library(TritonGPUIR
  Dialect.cpp
  Traits.cpp
  Types.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
)
