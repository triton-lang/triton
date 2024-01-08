add_triton_ut(
  NAME TestTritonAnalysis
  SRCS UtilityTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
)
