add_triton_library(TritonCPUTransforms
    ConvertDotOp/ConvertDotGeneric.cpp
    ConvertDotOp/ConvertDotToAMX.cpp
    ConvertDotProduct.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)
