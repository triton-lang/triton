add_triton_library(NVHopperTransforms
  WarpSpecialization.cpp
  WarpSpecialization/CodePartitionUtility.cpp
  WarpSpecialization/WSBuffer.cpp
  WarpSpecialization/WSCodePartition.cpp
  WarpSpecialization/WSLowerMem.cpp
  WarpSpecialization/WSSpecialize.cpp
  WarpSpecialization/WSTaskPartition.cpp
  WarpSpecialization/Utility.cpp

  DEPENDS
  NVHopperTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
