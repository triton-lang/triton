include_directories(${PROJECT_SOURCE_DIR}/third_party/nvidia/include)

add_triton_library(ProtonNVIDIAGPUToLLVM
    TargetInfo.cpp
    NvidiaPatternProtonGPUOpToLLVM.cpp
    ConvertProtonGPUToLLVM.cpp

    DEPENDS
    ProtonNvidiaGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    TritonNVIDIAGPUToLLVM
)
