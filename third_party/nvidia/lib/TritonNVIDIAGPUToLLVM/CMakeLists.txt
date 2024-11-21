add_triton_library(TritonNVIDIAGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv2OrV3.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/MMAv2.cpp
    DotOpToLLVM/WGMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    TMAToLLVM.cpp
    DecomposeUnsupportedConversions.cpp
    SPMDOpToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    ClusterOpsToLLVM.cpp
    PTXAsmFormat.cpp
    Utility.cpp
    UpcastMXFPToLLVM.cpp
    TargetInfo.cpp

    DEPENDS
    TritonNVIDIAGPUConversionPassIncGen
    NVGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
)
