add_mlir_library(TritonAMDGPUTestCommonUtils
  CommonUtils.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
