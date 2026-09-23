add_triton_library(TritonNVIDIAGPUToLLVM
    ConvertLayoutOpToLLVM.cpp
    ConvertWarpSpecializeToLLVM.cpp
    MemoryOpToLLVM.cpp
    DotOpToLLVM/MMAv2.cpp
    DotOpToLLVM/MMAv5.cpp
    DotOpToLLVM/WGMMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    Membar.cpp
    TritonGPUToLLVM.cpp
    TMAToLLVM.cpp
    SPMDOpToLLVM.cpp
    TensorMemoryToLLVM.cpp
    ClusterOpsToLLVM.cpp
    PTXAsmFormat.cpp
    Utility.cpp
    Fp4ToFpOpToLLVM.cpp
    TargetInfo.cpp
    Allocation.cpp

    DEPENDS
    TritonNVIDIAGPUConversionPassIncGen
    NVGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    GluonTransforms
    TritonAnalysis
    TritonGPUToLLVM
    TritonInstrumentIR
    TritonInstrumentToLLVM
    # MLIR's imported GPU conversion targets expose the entire GPU runtime in
    # their link interface, including the AMDGPU backend. Link the conversion
    # archives directly and spell the pieces used by these patterns explicitly.
    $<TARGET_FILE:MLIRGPUToNVVMTransforms>
    $<TARGET_FILE:MLIRMathToNVVM>
    $<TARGET_FILE:MLIRNVGPUToNVVM>
    $<TARGET_FILE:MLIRGPUToGPURuntimeTransforms>
    $<TARGET_FILE:MLIRGPUTransforms>
    $<TARGET_FILE:MLIRAsyncToLLVM>
    $<TARGET_FILE:MLIRAsyncDialect>
    $<TARGET_FILE:MLIRMemRefToLLVM>
    $<TARGET_FILE:MLIRVectorToLLVM>
    $<TARGET_FILE:MLIRMathToLLVM>
    $<TARGET_FILE:MLIRTargetLLVMIRExport>
    $<TARGET_FILE:MLIRLLVMToLLVMIRTranslation>
    $<TARGET_FILE:MLIRFuncToLLVM>
    $<TARGET_FILE:MLIRConvertToLLVMPass>
    $<TARGET_FILE:MLIRConvertToLLVMInterface>
    $<TARGET_FILE:MLIRVectorTransforms>
    $<TARGET_FILE:MLIRGPUUtils>
    $<TARGET_FILE:MLIRNVVMDialect>
    $<TARGET_FILE:MLIRNVGPUDialect>
    MLIRReconcileUnrealizedCasts
    NVGPUIR
    MLIRUBToLLVM
)
