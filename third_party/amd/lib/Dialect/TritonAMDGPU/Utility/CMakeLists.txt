add_triton_library(TritonAMDUtils
  CommonUtils.cpp
  TDMMergeUtility.cpp

  DEPENDS
  TritonAMDGPUTableGen
  TritonAMDGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)
