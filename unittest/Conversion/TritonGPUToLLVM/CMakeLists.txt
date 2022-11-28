add_triton_ut(
	NAME TestPtxAsmFormat
	SRCS PtxAsmFormatTest.cpp
	LIBS TritonGPUToLLVM
)
add_triton_ut(
	NAME TestGcnAsmFormat
	SRCS GcnAsmFormatTest.cpp
	LIBS TritonGPUToLLVM
)
