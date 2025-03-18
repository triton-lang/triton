add_triton_library(ProtonNVIDIAGPUToLLVM
    TargetInfo.cpp
    ConvertProtonGPUToLLVM.cpp
    ReadCounterOpToLLVM.cpp

    DEPENDS
    ProtonNvidiaGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    TritonNVIDIAGPUToLLVM
)
