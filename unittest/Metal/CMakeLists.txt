add_triton_ut(
  NAME ArgumentBinding
  SRCS ArgumentBindingTest.cpp
  LIBS
    TritonTools
)
add_triton_ut(
  NAME RuntimeTest
  SRCS RuntimeTest.cpp
  LIBS
    TritonTools
)
add_triton_ut(
  NAME PipelineConcurrency
  SRCS PipelineConcurrencyTest.cpp
  LIBS
    TritonTools
)
add_triton_ut(
  NAME DriverTest
  SRCS DriverTest.cpp
  LIBS
    TritonTools
    Python3::Module
)
