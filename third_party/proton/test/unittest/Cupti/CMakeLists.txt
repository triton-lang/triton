add_triton_ut(
  NAME CuptiMetricKernelQueue
  SRCS MetricKernelQueueTest.cpp
)

target_include_directories(CuptiMetricKernelQueue
  PRIVATE
    "${PROTON_SRC_DIR}/include"
)
