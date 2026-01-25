add_library(TritonTestAnalysis
  TestAlias.cpp
  TestAxisInfo.cpp
  TestAllocation.cpp
  TestBufferRegion.cpp
  TestMembar.cpp
)
target_link_libraries(TritonTestAnalysis PUBLIC MLIRPass TritonAnalysis TritonNvidiaGPUTransforms)
target_compile_options(TritonTestAnalysis PRIVATE ${TRITON_DISABLE_EH_RTTI_FLAGS})
