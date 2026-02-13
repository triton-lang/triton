add_triton_library(NVWSTransforms
  LowerAref.cpp
  LowerWarpGroup.cpp
  InsertAref.cpp
  Utilities.cpp
  AssignStagePhase.cpp
  InsertTmemAref.cpp
  HoistTmemStore.cpp

  DEPENDS
  NVWSTransformsIncGen

)
