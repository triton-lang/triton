add_triton_library(NVHopperTransforms
  WarpSpecialization.cpp
  WSTaskPartition.cpp
  Utility.cpp

  DEPENDS
  NVHopperTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
