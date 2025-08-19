add_triton_library(TritonMetalGPUIR
  Dialect.cpp
  Ops.cpp

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
