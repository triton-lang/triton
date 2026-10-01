add_triton_library(TritonGPUToLLVM
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/FMADotUtility.cpp
    AllocateSharedMemory.cpp
    AllocateSharedMemoryUtility.cpp
    AllocateWarpGroups.cpp
    AssertOpToLLVM.cpp
    CanonicalizeLLVMIR.cpp
    ControlFlowOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    Fp4ToFpOpToLLVMBase.cpp
    FuncOpToLLVM.cpp
    GatherOpToLLVM.cpp
    GlobalScratchMemoryAllocation.cpp
    HistogramOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    PrintOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp
    ViewOpToLLVM.cpp
    WarpSpecializeUtility.cpp

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRFuncToLLVM
    NVGPUIR
    TritonNVIDIAGPUToLLVMUtility
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
)

# The backend-specific MLIR GPU lowering dialects are only available when the
# respective backends are built.
if(TRITON_BUILD_NVIDIA_BACKEND)
  target_link_libraries(TritonGPUToLLVM PUBLIC MLIRGPUToNVVMTransforms)
endif()
if(TRITON_BUILD_AMD_BACKEND)
  target_link_libraries(TritonGPUToLLVM PUBLIC MLIRGPUToROCDLTransforms)
endif()
