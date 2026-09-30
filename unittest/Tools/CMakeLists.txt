add_triton_ut(
	NAME LinearLayout
	SRCS LayoutUtilsTest.cpp LinearLayoutTest.cpp
	LIBS TritonTools
)

add_triton_ut(
	NAME LLVMOptions
	SRCS LLVMOptionsTest.cpp
	LIBS TritonTools
)
