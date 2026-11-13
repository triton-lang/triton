add_triton_ut(
  NAME TestTritonAnalysis
  SRCS UtilityTest.cpp MembarCutTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
