add_triton_library(TritonCPUTransforms

  DEPENDS
  TritonCPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonAnalysis
  TritonIR
  TritonCPUIR
  MLIRTransformUtils
)
