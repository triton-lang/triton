add_triton_library(TritonNvidiaGPUToLLVM
    TritonNvidiaGPUToLLVMPass.cpp

    DEPENDS
    TritonNvidiaGPUConversionPassIncGen

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
    TritonNvidiaGPUIR
)
