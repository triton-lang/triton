include_directories(${PROJECT_SOURCE_DIR}/third_party/amd/include)

add_triton_library(ProtonAMDGPUToLLVM
    TargetInfo.cpp
    AMDPatternProtonGPUOpToLLVM.cpp
    AddSchedBarriers.cpp
    ConvertProtonGPUToLLVM.cpp

    DEPENDS
    ProtonAMDGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    TritonAMDGPUToLLVM
)
