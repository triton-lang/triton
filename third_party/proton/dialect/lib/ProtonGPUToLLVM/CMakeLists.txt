add_triton_library(ProtonGPUToLLVM
    RecordOpToLLVM.cpp
    ProtonGPUToLLVM.cpp
    AllocateProtonSharedMemory.cpp
    AddProtonKernelArg.cpp
    AllocateProtonGlobalScratchBuffer.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)
