add_triton_library(TritonProtonToLLVM
    InitScopeOpToLLVM.cpp
    RecordOpToLLVM.cpp

    LINK_LIBS PUBLIC
    ProtonIR
)
