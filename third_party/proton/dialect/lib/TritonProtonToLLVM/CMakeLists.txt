add_triton_library(TritonProtonToLLVM
    RecordOpToLLVM.cpp
    InitLocalBufferOpToLLVM.cpp
    AllocateProtonSMEMBuffer.cpp

    DEPENDS
    TritonProtonConversionPassIncGen
    
    LINK_LIBS PUBLIC
    ProtonIR
)
