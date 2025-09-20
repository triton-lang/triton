add_mlir_library(TritonAMDGPUTestAnalysis
  TestAMDRangeAnalysis.cpp
  TestAMDGPUMembar.cpp
  TestAxisInfo.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  ${triton_libs}
)
