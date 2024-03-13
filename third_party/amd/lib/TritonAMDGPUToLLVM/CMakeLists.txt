add_triton_library(TritonAMDGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM/WMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    GCNAsmFormat.cpp
    TritonGPUToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    DecomposeUnsupportedConversions.cpp
    SPMDOpToLLVM.cpp

    DEPENDS
    TritonAMDGPUConversionPassIncGen
    
    LINK_LIBS PUBLIC
    TritonGPUToLLVM
)

target_compile_definitions(TritonAMDGPUToLLVM PUBLIC USE_ROCM)
