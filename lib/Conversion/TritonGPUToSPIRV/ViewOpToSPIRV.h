#ifndef TRITON_VIEWOPTOSPIRV_H
#define TRITON_VIEWOPTOSPIRV_H

#include "TritonGPUToSPIRVBase.h"

using namespace mlir;
using namespace mlir::triton;

void populateViewOpToSPIRVPatterns(mlir::SPIRVTypeConverter &typeConverter,
                                   mlir::MLIRContext *context,
                                   mlir::RewritePatternSet &patterns,
                                   int numWarps,
                                   mlir::AxisInfoAnalysis &axisInfoAnalysis,
                                   const mlir::Allocation *allocation,
                                   mlir::Value smem,
                                   mlir::PatternBenefit benefit);

#endif
