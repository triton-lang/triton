#ifndef TRITON_CONVERSION_TRITONGPU_TO_SPIRV_LOAD_STORE_OP_H
#define TRITON_CONVERSION_TRITONGPU_TO_SPIRV_LOAD_STORE_OP_H

#include "TritonGPUToSPIRVBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateLoadStoreOpToSPIRVPatterns(
    mlir::SPIRVTypeConverter &typeConverter, mlir::MLIRContext *context, RewritePatternSet &patterns,
    int numWarps, AxisInfoAnalysis &axisInfoAnalysis,
    const Allocation *allocation, Value smem,
    ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
    PatternBenefit benefit);

#endif
