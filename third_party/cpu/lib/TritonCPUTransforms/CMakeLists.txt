add_triton_library(TritonCPUTransforms
    ConvertDotProduct.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)
