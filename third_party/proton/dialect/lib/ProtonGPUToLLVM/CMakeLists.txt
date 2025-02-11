add_triton_library(ProtonGPUToLLVM
    RecordOpToLLVM.cpp
    ProtonGPUToLLVM.cpp

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)
