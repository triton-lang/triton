#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir::triton::applegpu {

// Lower AppleMmaEncoding tt.dot → simdgroup_multiply_accumulate LLVM calls
std::unique_ptr<mlir::Pass> createConvertTritonAppleGPUToLLVMPass();

// Populate just the dot op patterns (for use in combined lowering passes)
void populateDotOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit = 1);

} // namespace mlir::triton::applegpu
