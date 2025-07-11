add_triton_library(GluonIR
  Dialect.cpp

  DEPENDS
  GluonTableGen
  GluonAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
