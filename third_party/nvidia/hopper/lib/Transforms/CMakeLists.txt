add_triton_library(NVHopperTransforms
  WarpSpecialization.cpp
  WarpSpecialization/CodePartitionUtility.cpp
  WarpSpecialization/PingPong.cpp
  WarpSpecialization/TaskIdPropagation.cpp
  WarpSpecialization/Utility.cpp
  WarpSpecialization/WSBuffer.cpp
  WarpSpecialization/WSCodePartition.cpp
  WarpSpecialization/WSDataPartition.cpp
  WarpSpecialization/WSLowerMem.cpp
  WarpSpecialization/WSLowerToken.cpp
  WarpSpecialization/WSSpecialize.cpp
  WarpSpecialization/WSTaskIdPropagate.cpp
  WarpSpecialization/WSTaskPartition.cpp

  DEPENDS
  NVHopperTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
