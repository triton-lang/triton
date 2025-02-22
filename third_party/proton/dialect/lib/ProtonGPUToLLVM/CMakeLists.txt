add_triton_library(ProtonGPUToLLVM
    RecordOpToLLVM.cpp
    ProtonGPUToLLVM.cpp
    GlobalScratchAllocOpToLLVM.cpp

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)
