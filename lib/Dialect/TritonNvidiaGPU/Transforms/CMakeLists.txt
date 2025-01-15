add_triton_library(TritonNvidiaGPUTransforms
  FenceInsertion.cpp
  PlanCTA.cpp
  TMALowering.cpp
  Utility.cpp

  DEPENDS
  TritonNvidiaGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
