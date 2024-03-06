add_triton_library(TritonAMDGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    GCNAsmFormat.cpp
    TritonGPUToLLVMPass.cpp
    Utility.cpp
    TargetInfo.cpp
    DecomposeUnsupportedConversions.cpp

    DEPENDS
    TritonAMDGPUConversionPassIncGen
    
    LINK_LIBS PUBLIC
    TritonGPUToLLVM
)

target_compile_definitions(TritonAMDGPUToLLVM PUBLIC USE_ROCM)

