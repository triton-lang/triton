add_triton_ut(
  NAME TestTritonAnalysis
  SRCS UtilityTest.cpp
  LIBS
    TritonAnalysis
    TritonAMDGPUIR
    TritonAMDUtils
    TritonIR
    TritonGPUIR
    TritonGPUToLLVM
    TritonGPUTransforms
    TritonInstrumentIR
    TritonNvidiaGPUIR
    TritonNvidiaGPUTransforms
    NVWSIR
    ProtonGPUIR
)
