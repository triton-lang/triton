add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    ViewSliceOpToLLVM.cpp

    DEPENDS
    TritonAMDGPUIR
)
