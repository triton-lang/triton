add_triton_library(NVHopperTransforms
  WarpSpecialization.cpp
  WarpSpecialization/WSTaskPartition.cpp
  WarpSpecialization/Utility.cpp

  DEPENDS
  NVHopperTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
