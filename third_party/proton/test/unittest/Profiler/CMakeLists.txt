add_triton_ut(
  NAME ProtonProfilerUnit
  SRCS
    ProfilerTest.cpp
    ${PROTON_SRC_DIR}/lib/Context/Context.cpp
    ${PROTON_SRC_DIR}/lib/Data/Data.cpp
    ${PROTON_SRC_DIR}/lib/Profiler/Profiler.cpp
)

target_include_directories(ProtonProfilerUnit
PRIVATE
  "${PROTON_COMMON_DIR}/include"
  "${PROTON_SRC_DIR}/include"
)
