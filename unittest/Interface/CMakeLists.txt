add_triton_ut(
  NAME TestTritonInterface
  SRCS MemoryEffectsTest.cpp
  LIBS
    TritonAMDGPUIR
    TritonAMDUtils
    TritonIR
    TritonGPUIR
    TritonGPUToLLVM
    TritonInstrumentIR
    TritonNvidiaGPUIR
    NVWSIR
    ProtonGPUIR
)
