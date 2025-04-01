include_directories(${PROJECT_SOURCE_DIR}/third_party/amd/include)

add_triton_library(ProtonNVIDIAGPUToLLVM
    TargetInfo.cpp
    ConvertProtonGPUToLLVM.cpp

    DEPENDS
    ProtonNvidiaGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    TritonNVIDIAGPUToLLVM
)
