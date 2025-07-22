add_triton_library(NVWSTransforms
  LowerAref.cpp
  LowerWarpGroup.cpp
  LowerSemaphore.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  MLIRTransformUtils
)
