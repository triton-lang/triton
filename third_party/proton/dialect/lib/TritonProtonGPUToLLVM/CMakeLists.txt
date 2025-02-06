add_triton_library(TritonProtonGPUToLLVM
    InitScopeOpToLLVM.cpp
    RecordOpToLLVM.cpp
    ProtonLoweringPass.cpp

    DEPENDS
    TritonProtonConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
)
