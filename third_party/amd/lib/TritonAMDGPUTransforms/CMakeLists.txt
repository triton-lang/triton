add_triton_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  OptimizeEpilogue.cpp
  RemoveLayoutConversions.cpp
  ReorderInstructions.cpp
  StreamPipeline.cpp
  MfmaGroup.cpp

  DEPENDS
  TritonAMDGPUTransformsIncGen
)

target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
target_compile_definitions(TritonAMDGPUTransforms PUBLIC USE_ROCM)
