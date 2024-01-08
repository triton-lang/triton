add_triton_ut(
	NAME TestPtxAsmFormat
	SRCS PTXAsmFormatTest.cpp
	LIBS TritonGPUToLLVM
)

add_triton_ut(
	NAME TestEmitIndices
	SRCS EmitIndicesTest.cpp DumpLayout.cpp
	LIBS TritonGPUIR TritonNvidiaGPUIR ${dialect_libs} ${conversion_libs}
)
