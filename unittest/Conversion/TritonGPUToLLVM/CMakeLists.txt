if ("nvidia" IN_LIST TRITON_CODEGEN_BACKENDS)
	add_triton_ut(
		NAME TestPtxAsmFormat
		SRCS PTXAsmFormatTest.cpp
		LIBS TritonGPUToLLVM TritonNVIDIAGPUToLLVM
	)

	add_triton_ut(
		NAME TestEmitIndicesNvidia
		SRCS EmitIndicesTest.cpp DumpLayout.cpp
		LIBS TritonGPUIR TritonNvidiaGPUIR TritonNVIDIAGPUToLLVM
		DEFS NVIDIA_TARGET=1
	)
endif()


if ("amd" IN_LIST TRITON_CODEGEN_BACKENDS)
	add_triton_ut(
		NAME TestEmitIndicesAMD
		SRCS EmitIndicesTest.cpp DumpLayout.cpp
		LIBS TritonGPUIR TritonAMDGPUToLLVM
		DEFS AMD_TARGET=1
	)
endif()
