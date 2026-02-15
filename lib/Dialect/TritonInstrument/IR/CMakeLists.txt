add_triton_library(TritonInstrumentIR
  Dialect.cpp
  FunctionBuilder.cpp
  Ops.cpp
  Utility.cpp

  DEPENDS
    TritonInstrumentTableGen

  LINK_LIBS PUBLIC
    MLIRIR
    TritonIR
    TritonGPUIR
)
