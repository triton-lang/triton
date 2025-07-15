add_triton_library(ProtonGPUTransforms
  ProtonGPUTransformsPass.cpp

  DEPENDS
  ProtonGPUTransformsIncGen
  LINK_LIBS PUBLIC
  ProtonGPUIR
)
