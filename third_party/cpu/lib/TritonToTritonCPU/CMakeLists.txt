add_triton_library(TritonToTritonCPU
    ConvertControlFlowOps.cpp
    ConvertDotOp.cpp
    ConvertElementwiseOps.cpp
    ConvertMemoryOps.cpp
    ConvertReductionOp.cpp
    ConvertPtrOps.cpp
    Pipeline.cpp
    TypeConverter.cpp

    DEPENDS
    TritonToTritonCPUPassIncGen

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRVectorDialect
)
