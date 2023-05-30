add_mlir_conversion_library(TritonGPUToLLVM
    TritonGPUToLLVM.cpp
    GCNAsmFormat.cpp
    PTXAsmFormat.cpp
    TritonGPUToLLVMPass.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv1.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv2.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMFMA.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MMAv1.cpp
    DotOpToLLVM/MMAv2.cpp
    DotOpToLLVM/MFMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    TritonGPUToLLVMPass.cpp
    GCNAsmFormat.cpp
    PTXAsmFormat.cpp
    ReduceOpToLLVM.cpp
    Utility.cpp
    TypeConverter.cpp
    ViewOpToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToLLVM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonGPUToLLVM

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUOps
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
)
