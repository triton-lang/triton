add_triton_library(TritonInstrumentIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
    TritonInstrumentTableGen

  LINK_LIBS PUBLIC
    MLIRIR
    TritonIR
    TritonGPUIR
)
