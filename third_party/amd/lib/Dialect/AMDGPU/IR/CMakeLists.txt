add_triton_library(AMDGPUIR
  Dialect.cpp

  DEPENDS
  AMDGPUTableGen
  AMDGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
