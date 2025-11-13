add_triton_library(TritonNvidiaGPUTransforms
  CheckMatmulTwoCTAs.cpp
  FenceInsertion.cpp
  InterleaveTMem.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeTMemLayouts.cpp
  PlanCTA.cpp
  PromoteLHSToTMem.cpp
  ProxFenceInsertion.cpp
  RemoveTMEMTokens.cpp
  TensorMemoryAllocation.cpp
  TMALowering.cpp
  TMAUtilities.cpp

  DEPENDS
  TritonNvidiaGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
