add_triton_library(ProtonGPUToLLVM
    TargetInfoBase.cpp
    PatternProtonGPUOpToLLVM.cpp
    AllocateProtonSharedMemory.cpp
    AddProtonKernelArg.cpp
    AllocateProtonGlobalScratchBuffer.cpp
    RecordOpToLLVM.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)

add_subdirectory(ProtonNvidiaGPUToLLVM)
add_subdirectory(ProtonAMDGPUToLLVM)
