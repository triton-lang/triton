add_triton_ut(
  NAME NvmmaSmemAttrsTest
  SRCS NvmmaSmemAttrsTest.cpp
  LIBS
    TritonAnalysis
    TritonGPUIR
    TritonNvidiaGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
