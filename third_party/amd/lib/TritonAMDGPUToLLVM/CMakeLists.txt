add_triton_library(TritonAMDGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    GCNAsmFormat.cpp
    TritonGPUToLLVMPass.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp
    ViewOpToLLVM.cpp
    # TensorPtrOpsToLLVM.cpp
    # ClusterOpsToLLVM.cpp
    # RegReallocOpToLLVM.cpp

    DEPENDS
    TritonAMDGPUConversionPassIncGen
)