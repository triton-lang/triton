add_triton_ut(
  NAME TestSwizzling
  SRCS SwizzleTest.cpp
  LIBS
    TritonTools
    LLVMSupport
    MLIRSupport
)
add_triton_ut(
  NAME Dialect
  SRCS DialectTest.cpp
  LIBS
    MLIRParser
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
add_triton_ut(
  NAME LinearLayoutConversions
  SRCS LinearLayoutConversionsTest.cpp
  LIBS
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)

add_triton_ut(
  NAME DumpLayoutTest
  SRCS DumpLayoutTest.cpp
  LIBS
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)
