add_mlir_dialect_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  AccelerateAMDMatmul.cpp
  Coalesce.cpp
  DecomposeConversions.cpp
  OptimizeDotOperands.cpp
  OptimizeEpilogue.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Prefetch.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  AMDReorderInstructions.cpp
  StreamPipeline.cpp
  DotSlicing.cpp
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
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
