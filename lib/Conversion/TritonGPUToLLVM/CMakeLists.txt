# Separate out PTX/GCN builders to avoid cyclic dependencies as TritonAnalysis
# depends on it.
set(LLVM_OPTIONAL_SOURCES
    PTXAsmFormat.cpp
  )

add_mlir_conversion_library(TritonGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv1.cpp
    ConvertLayoutOpToLLVM/SharedToDotOperandMMAv2.cpp
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/MMAv1.cpp
    DotOpToLLVM/MMAv2.cpp
    DotOpToLLVM/WGMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    HistogramOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    TritonGPUToLLVMPass.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp
    ViewOpToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    ClusterOpsToLLVM.cpp
    RegReallocOpToLLVM.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonGPUToLLVM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonGPUToLLVM

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    ASMBuilder
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    NVGPUIR
)

add_mlir_library(ASMBuilder
  PTXAsmFormat.cpp

  DEPENDS
  TritonTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
)
