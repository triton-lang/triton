add_triton_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  OptimizeEpilogue.cpp
  OptimizeFMADot.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp
  StreamPipelineV2.cpp
  MfmaGroup.cpp

  DEPENDS
  TritonAMDGPUTransformsIncGen
  TritonGPUIR
)

target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
