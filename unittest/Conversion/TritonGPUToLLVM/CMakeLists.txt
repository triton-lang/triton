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
if(WIN32)
# Set C++20 standard for TestEmitIndicesNvidia target
target_compile_features(TestEmitIndicesNvidia PRIVATE cxx_std_20)
endif()
if(NOT WIN32)
add_triton_ut(
	NAME TestEmitIndicesAMD
	SRCS EmitIndicesTest.cpp DumpLayout.cpp
	LIBS TritonGPUIR TritonAMDGPUToLLVM
	DEFS AMD_TARGET=1
)
endif()
