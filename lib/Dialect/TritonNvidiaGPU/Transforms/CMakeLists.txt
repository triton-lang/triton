add_mlir_dialect_library(TritonNvidiaGPUTransforms
  MaterializeLoadStore.cpp
  PlanCTA.cpp
  WSDecomposing.cpp
  WSFeasibilityChecking.cpp
  WSPipeline.cpp
  WSMaterialization.cpp
  Utility.cpp

  DEPENDS
  TritonNvidiaGPUTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
