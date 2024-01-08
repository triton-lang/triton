add_triton_ut(
  NAME TestTritonAnalysis
  SRCS UtilityTest.cpp
  LIBS
    ${TRITON_LIBRARIES}
    TritonAnalysis
    TritonIR
    TritonGPUIR
)
