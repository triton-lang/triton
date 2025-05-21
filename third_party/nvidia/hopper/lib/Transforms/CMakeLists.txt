add_triton_library(NVHopperTransforms
  WarpSpecialization.cpp
  WarpSpecialization/CodePartitionUtility.cpp
  WarpSpecialization/WSBuffer.cpp
  WarpSpecialization/WSCodePartition.cpp
  WarpSpecialization/WSLowerMem.cpp
  WarpSpecialization/WSSpecialize.cpp
  WarpSpecialization/Utility.cpp
  WarpSpecialization/WSDataPartition.cpp
  WarpSpecialization/WSTaskPartition.cpp

  DEPENDS
  NVHopperTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
