add_mlir_conversion_library(TritonToTritonGPUROCM
    TritonToTritonGPUPass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonToTritonGPUROCM
    ${PROJECT_BINARY_DIR}/include/triton/Conversion/TritonToTritonGPUROCM

    DEPENDS
    TritonToTritonGPUROCMConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUROCMIR
    TritonGPUROCMTransforms
)
