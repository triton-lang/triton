add_triton_library(TritonCPUToLLVM
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
