add_mlir_conversion_library(TritonGPUToLLVM
    TritonGPUToLLVM.cpp
    TritonGPUToLLVMPass.cpp
    PTXAsmFormat.cpp
    ConvertLayoutOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    ViewOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    DotOpToLLVM.cpp
    ReduceOpToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToLLVM

    DEPENDS
    TritonConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUOps
    MLIRGPUToNVVMTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
