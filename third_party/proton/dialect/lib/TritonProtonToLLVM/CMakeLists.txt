add_triton_library(TritonProtonToLLVM
    RecordOpToLLVM.cpp
    InitDeviceBufferOpToLLVM.cpp
    AllocateProtonDeviceBuffer.cpp

    DEPENDS
    TritonProtonConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
)
