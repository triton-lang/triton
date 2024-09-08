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
add_triton_ut(
	NAME LinearLayoutConversions
	SRCS LinearLayoutConversionsTest.cpp
	LIBS TritonGPUIR
)

add_triton_ut(
	NAME DumpLayoutTest
	SRCS DumpLayoutTest.cpp
	LIBS TritonGPUIR
)
