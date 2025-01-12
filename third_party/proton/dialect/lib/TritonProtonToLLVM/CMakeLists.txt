add_triton_library(TritonProtonToLLVM
    RecordOpToLLVM.cpp
    InitLocalBufferOpToLLVM.cpp

    LINK_LIBS PUBLIC
    ProtonIR
)
