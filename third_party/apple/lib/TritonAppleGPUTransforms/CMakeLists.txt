add_triton_library(TritonAppleGPUTransforms
  AppleMmaGroup.cpp
  AccelerateAppleMatmul.cpp  # TODO

  DEPENDS
  TritonAppleGPUDialectIncGen
  TritonAppleGPUAttrDefsIncGen
  TritonAppleGPUTransformsIncGen
  TritonGPUIR
)

target_include_directories(TritonAppleGPUTransforms PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAppleGPUTransforms PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/../../include)
