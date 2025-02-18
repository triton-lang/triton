add_triton_library(ProtonToProtonGPU
  ProtonLowering.cpp

  DEPENDS
  ProtonToProtonGPUIncGen
  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  ProtonIR
  ProtonGPUIR
)
