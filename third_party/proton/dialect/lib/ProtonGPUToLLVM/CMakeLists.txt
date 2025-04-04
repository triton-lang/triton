add_triton_library(ProtonGPUToLLVM
    AllocateProtonGlobalScratchBuffer.cpp
    AllocateProtonSharedMemory.cpp
    PatternProtonGPUOpToLLVM.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)

add_subdirectory(ProtonNvidiaGPUToLLVM)
add_subdirectory(ProtonAMDGPUToLLVM)
