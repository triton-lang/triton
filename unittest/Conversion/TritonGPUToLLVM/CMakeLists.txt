add_triton_ut(
  NAME TritonLLVMOpBuilderTest
  SRCS TritonLLVMOpBuilderTest.cpp
  LIBS
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
