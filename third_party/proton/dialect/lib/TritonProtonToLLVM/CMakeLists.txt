add_triton_library(TritonProtonToLLVM
    RecordOpToLLVM.cpp
    InitLocalBufferOpToLLVM.cpp

    DEPENDS
    TritonProtonConversionPassIncGen
    
    LINK_LIBS PUBLIC
    ProtonIR
)
