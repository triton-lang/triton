add_triton_library(NVWSTransforms
  AssignGroups.cpp
  PropagateGroups.cpp

  ArefCanonicalize.cpp
  ArefInsertion.cpp
  ArefCodeSplit.cpp
  ArefCopyElimination.cpp
  ArefCopyLowering.cpp
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
