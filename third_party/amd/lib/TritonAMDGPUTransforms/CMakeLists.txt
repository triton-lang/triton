add_triton_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  CanonicalizePointers.cpp
  OptimizeEpilogue.cpp
  ReorderInstructions.cpp
  StreamPipelineV2.cpp
  MfmaGroup.cpp
  inThreadTranspose.cpp

  DEPENDS
  TritonAMDGPUTransformsIncGen
  TritonGPUIR
)

target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
