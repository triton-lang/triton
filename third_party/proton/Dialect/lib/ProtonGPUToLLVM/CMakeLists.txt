add_triton_library(ProtonGPUToLLVM
    AllocateProtonSharedMemory.cpp
    PatternProtonGPUOpToLLVM.cpp
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
