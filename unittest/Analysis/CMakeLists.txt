add_triton_ut(
  NAME TestTritonAnalysis
  SRCS UtilityTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)

add_triton_ut(
  NAME TestTritonCallGraph
  SRCS CallGraphTest.cpp
  LIBS
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    MLIRParser
)
