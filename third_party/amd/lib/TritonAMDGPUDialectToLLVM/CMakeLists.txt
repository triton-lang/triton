add_triton_library(TritonAMDGPUDialectToLLVM
    TritonAMDGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    TransposeInRegistersOpToTTG.cpp

    DEPENDS
    TritonAMDGPUIR
)
