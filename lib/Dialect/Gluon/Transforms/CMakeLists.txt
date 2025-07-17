add_triton_library(GluonTransforms
  Canonicalize.cpp
  Inline.cpp
  ResolveAutoEncodings.cpp

  DEPENDS
  GluonTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  GluonIR
  MLIRTransformUtils
)
