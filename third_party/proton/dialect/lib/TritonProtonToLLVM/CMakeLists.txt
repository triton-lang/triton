add_triton_library(TritonProtonToLLVM
    InitScopeOpToLLVM.cpp
    RecordOpToLLVM.cpp
    ProtonLoweringPass.cpp

    DEPENDS
    TritonProtonConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
)
