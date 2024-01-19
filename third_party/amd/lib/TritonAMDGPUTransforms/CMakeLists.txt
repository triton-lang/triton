add_triton_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  DecomposeConversions.cpp
  OptimizeEpilogue.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp

  DEPENDS
  TritonAMDGPUTransformsIncGen
)
