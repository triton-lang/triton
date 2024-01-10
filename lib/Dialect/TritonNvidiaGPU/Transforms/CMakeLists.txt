add_triton_library(TritonNvidiaGPUTransforms
  MaterializeLoadStore.cpp
  PlanCTA.cpp
  WSDecomposing.cpp
  WSFeasibilityChecking.cpp
  WSPipeline.cpp
  WSMutex.cpp
  WSMaterialization.cpp
  WSFixupMissingAttrs.cpp
  FenceInsertion.cpp
  RewriteTensorPointer.cpp
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
