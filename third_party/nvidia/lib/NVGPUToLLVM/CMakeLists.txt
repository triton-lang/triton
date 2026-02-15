add_triton_library(NVGPUToLLVM
    NVGPUToLLVMPass.cpp

    DEPENDS
    NVGPUConversionPassIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    NVGPUIR
)
