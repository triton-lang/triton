add_triton_ut(
	NAME TestSwizzling
	SRCS SwizzleTest.cpp
	LIBS ${TRITON_LIBRARUES} TritonGPUIR TritonNvidiaGPUIR  ${dialect_libs} ${conversion_libs}
)
