add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    InThreadTransposeOpToTTG.cpp
    ConcatOpToLLVM.cpp
    ScaledDowncastToLLVM.cpp
    ScaledUpcastToLLVM.cpp

    DEPENDS
    TritonAMDGPUIR
)
