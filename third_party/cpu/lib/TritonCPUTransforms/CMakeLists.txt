add_triton_library(TritonCPUTransforms
    ConvertDotProduct.cpp
    ConvertDotToAMX.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)
