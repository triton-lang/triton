add_triton_library(NVGToLLVM
    NVGToLLVMPass.cpp

    DEPENDS
    NVGConversionPassIncGen

    LINK_LIBS PUBLIC
    NVGIR
)
