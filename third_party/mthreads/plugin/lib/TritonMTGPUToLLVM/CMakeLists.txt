add_triton_library(TritonMTGPUToLLVM
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    SPMDOpToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    MUSATranslation.cpp
    FuncOpToLLVM.cpp
    BuiltinFuncToLLVM.cpp

    DEPENDS
    TritonMTGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRMTGPUDialect
    MLIRMTGPUToLLVMIRTranslation
    MLIRGPUToMTGPUTransforms
    TritonGPUToLLVM
)
