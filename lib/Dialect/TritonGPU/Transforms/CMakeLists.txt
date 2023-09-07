add_mlir_dialect_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  Coalesce.cpp
  DecomposeConversions.cpp
  OptimizeDotOperands.cpp
  Pipeline.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp
  TritonGPUConversion.cpp
  Utility.cpp

  DEPENDS
  TritonGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonGPUIR
)
