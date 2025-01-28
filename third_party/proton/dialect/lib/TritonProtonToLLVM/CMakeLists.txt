add_triton_library(TritonProtonToLLVM
    RecordOpToLLVM.cpp
    BufferAllocOpToLLVM.cpp
    AllocateProtonDeviceBuffer.cpp

    DEPENDS
    TritonProtonConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
)
