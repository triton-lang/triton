add_triton_library(GluonTransforms
  ResolveAutoEncodings.cpp

  DEPENDS
  GluonTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  GluonIR
  MLIRTransformUtils
)
