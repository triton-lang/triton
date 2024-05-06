add_triton_library(TritonToTritonCPU
    TritonCPUConversion.cpp
    TritonToTritonCPUPass.cpp

    DEPENDS
    TritonConversionToCPUPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonCPUIR
    TritonCPUTransforms
)
