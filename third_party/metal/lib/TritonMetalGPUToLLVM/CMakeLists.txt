add_triton_library(TritonMetalGPUToLLVM
    FuncOpToLLVM.cpp
    MembarUtility.cpp
    MemoryOpToLLVM.cpp
    TargetInfo.cpp
    TritonGPUToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    Utility.cpp
    GPUIdxOpsToLLVM.cpp
    SPMDOpToLLVM.cpp
    DotOpToLLVM.cpp
    SimdgroupAsyncCopyOpToLLVM.cpp
    SimdgroupMMAOpToLLVM.cpp
    SimdgroupStoreOpToLLVM.cpp
    SimdgroupWaitOpToLLVM.cpp
    DotOpToLLVM/FMA.cpp
    DotOpToLLVM/SimdgroupMatmul.cpp

    DEPENDS
    TritonMetalGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonMetalAnalysis
)

target_include_directories(TritonMetalGPUToLLVM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonMetalGPUToLLVM PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)