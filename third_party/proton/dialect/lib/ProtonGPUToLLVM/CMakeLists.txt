add_triton_library(ProtonGPUToLLVM
    InitScopeOpToLLVM.cpp
    RecordOpToLLVM.cpp
    ProtonLoweringPass.cpp

    DEPENDS
    ProtonConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
)
