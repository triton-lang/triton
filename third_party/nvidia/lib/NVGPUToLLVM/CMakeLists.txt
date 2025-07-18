add_triton_library(NVGPUToLLVM
    NVGPUToLLVMPass.cpp

    DEPENDS
    NVGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    NVGPUIR
)
