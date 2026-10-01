# Backend-agnostic LLVM-IR helpers (PTX builder utilities and NVGPU layout
# helpers) used by core passes such as TritonInstrumentToLLVM. These only
# build LLVM IR and do not require the NVPTX target, so they are built even
# when the NVIDIA backend is disabled.
add_triton_library(TritonNVIDIAGPUToLLVMUtility
    PTXAsmFormat.cpp
    Utility.cpp

    DEPENDS
    NVGPUAttrDefsIncGen
)

if(TRITON_BUILD_NVIDIA_BACKEND)
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
    MLIRReconcileUnrealizedCasts
    MLIRMathToLLVM
    TritonNVIDIAGPUToLLVMUtility
    MLIRUBToLLVM
)
endif()
