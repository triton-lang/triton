add_triton_library(TritonCPUToLLVM
    ControlFlowOpToLLVM.cpp
    CPUTargetInfo.cpp
    FuncOpToLLVM.cpp
    PrintOpToLLVM.cpp
    SPMDOpToLLVM.cpp
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
