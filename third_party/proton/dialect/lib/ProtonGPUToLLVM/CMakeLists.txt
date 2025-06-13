add_triton_library(ProtonGPUToLLVM
    AllocateProtonGlobalScratchBuffer.cpp
    AllocateProtonSharedMemory.cpp
    PatternProtonGPUOpToLLVM.cpp
    AddSchedBarriers.cpp
    Utility.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)

add_subdirectory(ProtonNvidiaGPUToLLVM)
add_subdirectory(ProtonAMDGPUToLLVM)
