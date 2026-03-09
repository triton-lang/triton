add_triton_library(TritonAppleGPUToLLVM
  Dialect.cpp
  AppleMmaLayoutConversions.cpp
  DotOpToLLVM.cpp
  ConvertTritonAppleGPUToLLVM.cpp

  DEPENDS
  TritonAppleGPUDialectIncGen
  TritonAppleGPUAttrDefsIncGen
  TritonGPUIR
  MLIRLLVMDialect
)

target_include_directories(TritonAppleGPUToLLVM PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAppleGPUToLLVM PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/../../include)
