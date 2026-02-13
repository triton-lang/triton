add_triton_library(ProtonGPUToLLVM
    AllocateProtonGlobalScratchBuffer.cpp
    AllocateProtonSharedMemory.cpp
    PatternProtonGPUOpToLLVM.cpp
    Utility.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen
)

add_subdirectory(ProtonNvidiaGPUToLLVM)
add_subdirectory(ProtonAMDGPUToLLVM)
