add_triton_library(TritonCPUToLLVM
    AtomicOpsToLLVM.cpp
    FuncOpToLLVM.cpp
    GetProgramIdOpToLLVM.cpp
    LowerMultiReduction.cpp
    MemoryOpToLLVM.cpp
    Pipeline.cpp
    TypeConverter.cpp

    DEPENDS
    TritonCPUToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRVectorToLLVMPass
)
