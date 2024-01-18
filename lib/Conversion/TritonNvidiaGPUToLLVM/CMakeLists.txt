add_triton_library(TritonNvidiaGPUToLLVM
    TritonNvidiaGPUToLLVMPass.cpp
    # DotOpToLLVM/WGMMA.cpp
    # ClusterOpsToLLVM.cpp
    # RegReallocOpToLLVM.cpp
    # TensorPtrOpsToLLVM.cpp
    # BarrierOpToLLVM.cpp

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
