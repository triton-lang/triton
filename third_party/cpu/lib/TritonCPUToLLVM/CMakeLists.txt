add_triton_library(TritonCPUToLLVM
    AtomicOpsToLLVM.cpp
    DebugOpsToLLVM.cpp
    FuncOpToLLVM.cpp
    GetProgramIdOpToLLVM.cpp
    LowerMultiReduction.cpp
    MathToLibmvec.cpp
    MemoryOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp

    DEPENDS
    TritonCPUToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorToLLVMPass
)
