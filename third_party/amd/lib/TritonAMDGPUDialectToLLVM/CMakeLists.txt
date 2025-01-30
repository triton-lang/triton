add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    ConcatOpToLLVM.cpp

    DEPENDS
    TritonAMDGPUIR
)
