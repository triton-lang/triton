add_triton_library(TritonNvidiaGPUTransforms
  FenceInsertion.cpp
  MMALowering.cpp
  OptimizeDescriptorEncoding.cpp
  OptimizeTMemSubtiling.cpp
  PlanCTA.cpp
  PromoteLHSToTMem.cpp
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
