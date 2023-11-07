add_triton_ut(
	NAME TestSwizzling
	SRCS SwizzleTest.cpp VerifierTest.cpp
	LIBS TritonGPUIR TritonNvidiaGPUIR  ${dialect_libs} ${conversion_libs}
)
