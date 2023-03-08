#ifndef TRITON_TRITONGPUTOSPIRV_H
#define TRITON_TRITONGPUTOSPIRV_H

#include "TritonGPUToSPIRVBase.h"

void populateTritonGPUToSPIRVPatterns(
        mlir::SPIRVTypeConverter &typeConverter, mlir::MLIRContext *context,
        mlir::RewritePatternSet &patterns, int numWarps,
        mlir::AxisInfoAnalysis &axisInfoAnalysis,
        const mlir::Allocation *allocation, mlir::Value smem,
        ConvertTritonGPUOpToSPIRVPatternBase::IndexCacheInfo &indexCacheInfo,
        mlir::PatternBenefit benefit);

#endif
