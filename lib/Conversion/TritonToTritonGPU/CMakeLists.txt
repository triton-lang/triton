add_mlir_conversion_library(TritonToTritonGPU
    TritonToTritonGPU.cpp    

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/triton/Conversion/TritonToTritonGPU

    DEPENDS
    TritonConversionPassIncGen

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonIR
    TritonGPUIR
    TritonGPUConversion
)
