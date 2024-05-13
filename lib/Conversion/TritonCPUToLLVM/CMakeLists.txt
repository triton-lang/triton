add_triton_library(TritonCPUToLLVM
    ControlFlowOpToLLVM.cpp
    FuncOpToLLVM.cpp
    TypeConverter.cpp
    TritonCPUToLLVM.cpp

    DEPENDS
    TritonCPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonAnalysis
    TritonIR
    TritonCPUIR
    TritonCPUTransforms
)
