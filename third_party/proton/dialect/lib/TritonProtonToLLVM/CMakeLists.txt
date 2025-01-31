add_triton_library(TritonProtonToLLVM
    RecordOpToLLVM.cpp
    BufferAllocOpToLLVM.cpp
    ProtonLoweringPass.cpp

    DEPENDS
    TritonProtonConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
)
