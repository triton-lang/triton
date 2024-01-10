add_triton_ut(
	NAME TestSwizzling
	SRCS SwizzleTest.cpp
	LIBS TritonGPUIR TritonNvidiaGPUIR  ${dialect_libs} ${conversion_libs} ${triton_libs}
)
