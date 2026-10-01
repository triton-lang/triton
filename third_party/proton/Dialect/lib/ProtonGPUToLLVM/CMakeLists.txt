add_triton_library(ProtonGPUToLLVM
    AllocateProtonSharedMemory.cpp
    PatternProtonGPUOpToLLVM.cpp
    Utility.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)

if(TRITON_BUILD_NVIDIA_BACKEND)
  add_subdirectory(ProtonNvidiaGPUToLLVM)
endif()
if(TRITON_BUILD_AMD_BACKEND)
  add_subdirectory(ProtonAMDGPUToLLVM)
endif()
