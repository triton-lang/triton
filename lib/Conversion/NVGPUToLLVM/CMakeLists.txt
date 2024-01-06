add_mlir_conversion_library(NVGPUToLLVM
    NVGPUToLLVMPass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/NVGPUToLLVM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/NVGPUToLLVM

    DEPENDS
    NVGPUConversionPassIncGen

    LINK_COMPONENTS
    Core
)
