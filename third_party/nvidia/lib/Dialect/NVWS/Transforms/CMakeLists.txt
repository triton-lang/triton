add_triton_library(NVWSTransforms
  AssignGroups.cpp
  PropagateGroups.cpp
  LowerAref.cpp
  LowerWarpGroup.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  MLIRTransformUtils
)
