add_triton_library(ProtonIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  ProtonTableGen
  ProtonAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)

#target_include_directories(TritonProtonTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include/)
#target_include_directories(TritonProtonTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../../../include)

