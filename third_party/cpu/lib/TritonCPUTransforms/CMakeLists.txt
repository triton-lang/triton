add_triton_library(TritonCPUTransforms
    ConvertDotOp/ConvertDotGeneric.cpp
    ConvertDotOp/ConvertDotToAMX.cpp
    Canonicalize.cpp
    ConvertDotProduct.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)
