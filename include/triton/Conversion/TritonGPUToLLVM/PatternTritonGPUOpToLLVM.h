#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVMBase.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

using ::mlir::triton::gpu::BlockedEncodingAttr;

using ClampFOpFunction = std::function<SmallVector<Value>(
    ClampFOp, ConversionPatternRewriter &, Type, MultipleOperandsRange,
    Location, int)>;

namespace SharedToDotOperandFMA {
Value convertLayout(int opIdx, Value val, Value llVal,
                    BlockedEncodingAttr dLayout, Value thread, Location loc,
                    const LLVMTypeConverter *typeConverter,
                    ConversionPatternRewriter &rewriter);
}
LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);
namespace mlir {
namespace triton {

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, PatternBenefit benefit);

void populateMemoryOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

void populateAssertOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   PatternBenefit benefit);

void populateMakeRangeOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit);

void populateViewOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

void populateMinMaxFOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                    bool hwNanPropagationSupported,
                                    PatternBenefit benefit);

void populateClampFOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis,
    std::function<bool(ClampFOp, int)> optimizedClampPatternFound,
    ClampFOpFunction emitOptimization, ClampFOpFunction emitDefault,
    int computeCapability, PatternBenefit benefit);
} // namespace triton
} // namespace mlir

#endif
