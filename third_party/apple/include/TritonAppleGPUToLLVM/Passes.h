#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir::triton::applegpu {

// Lower AppleMmaEncoding tt.dot → simdgroup_multiply_accumulate LLVM calls
std::unique_ptr<mlir::Pass> createConvertTritonAppleGPUToLLVMPass();

// Lower remaining gpu.thread_id / gpu.block_dim → air intrinsics / constants
std::unique_ptr<mlir::Pass> createLowerGPUToAirPass();

// Populate just the dot op patterns (for use in combined lowering passes)
void populateDotOpToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit = 1);

// Populate load/store/addptr patterns
void populateLoadStoreToLLVMPatterns(
    mlir::LLVMTypeConverter &typeConverter,
    mlir::RewritePatternSet &patterns,
    mlir::PatternBenefit benefit = 1);

// Register all Apple GPU → LLVM passes with the MLIR pass registry
// (for use with triton-opt / mlir-opt command line tools).
void registerTritonAppleGPUToLLVMPasses();

} // namespace mlir::triton::applegpu
