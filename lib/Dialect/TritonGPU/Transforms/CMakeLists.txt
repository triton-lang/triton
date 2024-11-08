add_triton_library(TritonGPUTransforms
  AccelerateMatmul.cpp
  Coalesce.cpp
  F32DotTC.cpp
  CombineTensorSelectAndIf.cpp
  ReduceDataDuplication.cpp
  OptimizeAccumulatorInit.cpp
  OptimizeDotOperands.cpp
  OptimizeThreadLocality.cpp
  Pipeliner/MatmulLoopPipeline.cpp
  Pipeliner/OuterLoopPipeline.cpp
  Pipeliner/PipelineExpander.cpp
  Pipeliner/PipelineScheduler.cpp
  Pipeliner/TestPipelineAssignLatencies.cpp
  Pipeliner/SoftwarePipeliner.cpp
  Pipeliner/TMAStoresPipeline.cpp
  Pipeliner/PipeliningUtility.cpp
  Pipeliner/Schedule.cpp
  Prefetch.cpp
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
  TritonNvidiaGPUIR
  MLIRTransformUtils
)
