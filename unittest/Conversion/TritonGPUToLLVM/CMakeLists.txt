add_triton_ut(
	NAME TestPtxAsmFormat
	SRCS PTXAsmFormatTest.cpp
	LIBS TritonGPUToLLVM
)
<<<<<<< HEAD
add_triton_ut(
	NAME TestGcnAsmFormat
	SRCS GcnAsmFormatTest.cpp
	LIBS TritonGPUToLLVM
=======

add_triton_ut(
	NAME TestEmitIndices
	SRCS EmitIndicesTest.cpp DumpLayout.cpp
	LIBS TritonGPUIR TritonNvidiaGPUIR ${dialect_libs} ${conversion_libs}
>>>>>>> 36fc54b6f28168d3644808bfe299f1ba06a36272
)
