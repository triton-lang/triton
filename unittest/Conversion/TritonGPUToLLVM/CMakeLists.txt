add_triton_ut(
	NAME TestPtxAsmFormat
	SRCS PTXAsmFormatTest.cpp
	LIBS ${TRITON_LIBRARIES} TritonGPUToLLVM
)

add_triton_ut(
	NAME TestEmitIndices
	SRCS EmitIndicesTest.cpp DumpLayout.cpp
	LIBS ${TRITON_LIBRARIES} TritonGPUIR TritonNvidiaGPUIR ${dialect_libs} ${conversion_libs}
)
