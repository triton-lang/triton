add_triton_library(TritonCPUTransforms
    ConvertDotOp/ConvertDotCommon.cpp
    ConvertDotOp/ConvertDotGeneric.cpp
    ConvertDotOp/ConvertDotToAMX.cpp
    ConvertDotOp/ConvertDotToFMA.cpp
    ConvertDotOp/ConvertDotOpToUkernelOps.cpp
    ConvertDotOp/ConvertDotToNanokernel.cpp
    Canonicalize.cpp
    ConvertDotProduct.cpp
    ConvertUnsupportedOps.cpp
    ConvertVectorContractToArmNeonBFMMLA.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp
    UnrollAndReorderElementwiseOps.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRArithDialect
    MLIRArmNeonDialect
    MLIRTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
)

if (dnnl_FOUND)
  target_link_libraries(TritonCPUTransforms PRIVATE DNNL::dnnl)
endif()
