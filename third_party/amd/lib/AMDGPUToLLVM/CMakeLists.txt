add_triton_library(AMDGPUToLLVM
    AMDGPUToLLVMPass.cpp

    DEPENDS
    AMDGPUConversionPassIncGen
    AMDGPUIR
)
