add_triton_library(NVWSTransforms
  LowerAref.cpp
  LowerWarpGroup.cpp
  InsertAref.cpp
  Utilities.cpp
  AssignStagePhase.cpp
  InsertTmemAref.cpp

  DEPENDS
  NVWSTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  NVWSIR
  MLIRTransformUtils
)
