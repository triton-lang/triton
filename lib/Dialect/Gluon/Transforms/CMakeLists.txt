add_triton_library(GluonTransforms
  Canonicalize.cpp
  Inline.cpp
  ResolveAutoEncodings.cpp
  SimplifyControlFlow.cpp
  InferCoalescedEncodings.cpp
  InferLayoutUtils.cpp

  DEPENDS
  GluonTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  GluonIR
  MLIRTransformUtils
)
