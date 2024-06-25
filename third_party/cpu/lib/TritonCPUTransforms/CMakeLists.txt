add_triton_library(TritonCPUTransforms
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen
)
