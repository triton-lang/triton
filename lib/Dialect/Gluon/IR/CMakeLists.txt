add_triton_library(GluonIR
  Dialect.cpp

  DEPENDS
  GluonTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
