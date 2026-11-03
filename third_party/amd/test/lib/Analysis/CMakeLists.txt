add_library(TritonAMDGPUTestAnalysis
  TestAMDRangeAnalysis.cpp
  TestAMDGPUMembar.cpp
  TestAxisInfo.cpp
)
add_dependencies(TritonAMDGPUTestAnalysis
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
)
target_link_libraries(TritonAMDGPUTestAnalysis MLIRPass)
target_compile_options(TritonAMDGPUTestAnalysis PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
