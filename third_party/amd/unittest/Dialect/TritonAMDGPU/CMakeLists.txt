add_triton_ut(
  NAME TestAMDTargetFeatures
  SRCS TargetFeaturesTest.cpp
  LIBS
    TritonAMDGPUIR
    TritonAMDGPUToLLVM
    TritonAMDGPUTransforms
    TritonAMDGPUDialectToLLVM
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    TritonNVIDIAGPUToLLVM
    MLIRUBToLLVM
)
