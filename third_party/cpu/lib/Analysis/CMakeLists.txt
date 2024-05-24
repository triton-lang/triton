add_triton_library(TritonCPUAnalysis
  TensorPtrShapeInfo.cpp

  DEPENDS
  TritonCPUTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  TritonIR
  TritonCPUIR
)
