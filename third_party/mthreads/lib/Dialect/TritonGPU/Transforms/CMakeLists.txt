add_triton_library(TritonGPUTransforms
  Coalesce.cpp
  CombineTensorSelectAndIf.cpp
  ReduceDataDuplication.cpp
  OptimizeThreadLocality.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  Utility.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
