add_triton_library(ProtonGPUToLLVM
    RecordOpToLLVM.cpp
    ProtonGPUToLLVM.cpp
    ModuleOpToLLVM.cpp
    GlobalScratchAllocOpToLLVM.cpp
    AllocateProtonSharedMemory.cpp
    AddProtonKernelArg.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)
