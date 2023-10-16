add_mlir_dialect_library(TritonGPUROCMTransforms
  AccelerateMatmul.cpp
  Coalesce.cpp
  DecomposeConversions.cpp
  OptimizeDotOperands.cpp
  OptimizeEpilogue.cpp
  Pipeline.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp
  TritonGPUConversion.cpp
  Utility.cpp

  DEPENDS
  TritonGPUROCMTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysisROCM
  TritonIR
  TritonGPUROCMIR
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
