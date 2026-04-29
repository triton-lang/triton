add_triton_library(TritonMetalGPUTransforms
  AccelerateMetalMatmul.cpp
  PrepareSimdgroupMatmul.cpp
  InjectTensorStrideArgs.cpp

  DEPENDS
  TritonMetalGPUIR
  TritonMetalGPUTransformsIncGen
  TritonGPUIR
)

target_include_directories(TritonMetalGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonMetalGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
