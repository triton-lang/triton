add_triton_library(TritonGPUToLLVM
    TypeConverter.cpp
    Utility.cpp
    ElementwiseOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    AssertOpToLLVM.cpp
    ViewOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    NVGPUIR
)
