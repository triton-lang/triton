add_triton_library(ProtonTransforms
  ProtonLowering.cpp

  DEPENDS
  ProtonTransformsIncGen
  LINK_LIBS PUBLIC
  MLIRTransforms
  MLIRTransformUtils
  TritonIR
  TritonGPUIR
  TritonNvidiaGPUIR
  ProtonIR
)
