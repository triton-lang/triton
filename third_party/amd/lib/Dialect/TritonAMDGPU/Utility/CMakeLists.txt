add_triton_library(TritonAMDUtils
  CommonUtils.cpp
  TDMCopyFuseUtility.cpp

  DEPENDS
  TritonAMDGPUTableGen
  TritonAMDGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
