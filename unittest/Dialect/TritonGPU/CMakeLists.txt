add_triton_ut(
	NAME TestSwizzling
	SRCS SwizzleTest.cpp
	LIBS TritonGPUIR TritonNvidiaGPUIR
)
add_triton_ut(
	NAME Dialect
	SRCS DialectTest.cpp
	LIBS TritonGPUIR
)
