add_triton_library(TritonNvidiaGPUTransforms
  FenceInsertion.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeTMemSubtiling.cpp
  PlanCTA.cpp
  PromoteLHSToTMem.cpp
  RemoveTMEMTokens.cpp
  TensorMemoryAllocation.cpp
  TMALowering.cpp
  TMAUtilities.cpp
  TtngWgToArefIf.cpp
  TtngWgToTtgWs.cpp

  DEPENDS
  TritonNvidiaGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
  MLIRTransformUtils
  NVWSIR
  NVWSTransforms
)
