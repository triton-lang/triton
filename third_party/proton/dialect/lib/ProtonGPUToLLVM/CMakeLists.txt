add_triton_library(ProtonGPUToLLVM
    InitScopeOpToLLVM.cpp
    RecordOpToLLVM.cpp
    ProtonGPUToLLVM.cpp

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
)
