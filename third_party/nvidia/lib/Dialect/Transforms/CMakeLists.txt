add_triton_library(NVTransforms
  WarpSpecialization.cpp

  DEPENDS
  NVTransformsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  MLIRTransformUtils
)
