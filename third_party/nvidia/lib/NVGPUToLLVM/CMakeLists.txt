add_triton_library(NVGPUToLLVM
    NVGPUToLLVMPass.cpp

    DEPENDS
    NVGPUConversionPassIncGen
)

target_compile_definitions(NVGPUToLLVM PUBLIC USE_CUDA)
