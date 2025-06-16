add_triton_library(NVWSTransforms
  AssignGroups.cpp
  PropagateGroups.cpp

  ArefCanonicalize.cpp
  ArefTMEMInsertion.cpp
  ArefInsertion.cpp
  ArefCodeSplit.cpp
  ArefOptimize.cpp
  ArefAsyncOps.cpp
  ArefDepth.cpp
  ArefIndex.cpp
  ArefLowering.cpp
  FMHAMathLoopPipeline.cpp
  BlackwellFA.cpp

  LowerAref.cpp
  LowerWarpGroup.cpp

  Utility.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  MLIRTransformUtils
)
