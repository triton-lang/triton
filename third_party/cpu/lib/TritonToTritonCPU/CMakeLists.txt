add_triton_library(TritonToTritonCPU
    ConvertAtomicOps.cpp
    ConvertControlFlowOps.cpp
    ConvertDebugOps.cpp
    ConvertDotOp.cpp
    ConvertElementwiseOps.cpp
    ConvertElemManipOps.cpp
    ConvertHistogramOp.cpp
    ConvertMemoryOps.cpp
    ConvertPtrOps.cpp
    ConvertReductionOp.cpp
    ConvertScanOp.cpp
    TypeConverter.cpp

    DEPENDS
    TritonToTritonCPUPassIncGen

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRVectorDialect
)
