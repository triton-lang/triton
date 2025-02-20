add_triton_library(ProtonToProtonGPU
  ProtonToProtonGPUPass.cpp

  DEPENDS
  ProtonToProtonGPUIncGen
  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  ProtonIR
  ProtonGPUIR
)
