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
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp

    DEPENDS
    TritonAMDGPUConversionPassIncGen
    
    LINK_LIBS PUBLIC
    TritonGPUToLLVM
)