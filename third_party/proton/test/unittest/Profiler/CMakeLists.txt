add_triton_ut(
  NAME ProtonCorrelation
  SRCS CorrelationTest.cpp
)

target_include_directories(ProtonCorrelation
PRIVATE
  "${PROTON_COMMON_DIR}/include"
  "${PROTON_SRC_DIR}/include"
)
