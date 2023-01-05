
add_triton_ut(
	NAME TestSwizzling
	SRCS SwizzleTest.cpp
	LIBS TritonGPUIR  ${dialect_libs} ${conversion_libs}
)