add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    InThreadTransposeOpToTTG.cpp
    ConcatOpToLLVM.cpp
    Utility.cpp
    ScaledUpcastToLLVM.cpp

    DEPENDS
    TritonAMDGPUIR
)
