add_triton_library(TritonAMDGPUTransforms
  AccelerateAMDMatmul.cpp
  BlockPingpong.cpp
  CanonicalizePointers.cpp
  CoalesceAsyncCopy.cpp
  ConvertToBufferOps.cpp
  ConvertToTensorOps.cpp
  OptimizeEpilogue.cpp
  OptimizeDotOperands.cpp
  HoistLayoutConversions.cpp
  SinkLayoutConversions.cpp
  PrepareIfCombining.cpp
  ReorderInstructions.cpp
  Pipeline.cpp
  ScheduleLoops.cpp
  LowerLoops.cpp
  MfmaGroup.cpp
  WmmaGroup.cpp
  InThreadTranspose.cpp
  FoldTrueCmpIOp.cpp
  UpdateAsyncWaitCount.cpp
  Utility.cpp
  WarpPipeliner.cpp

  DEPENDS
  TritonAMDGPUIR
  TritonAMDGPUTransformsIncGen
  TritonGPUIR
  TritonAMDUtils
  TritonAMDAnalysis
)

target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonAMDGPUTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
