#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
namespace {

using namespace mlir;
using namespace mlir::triton;

using ::mlir::LLVM::getSharedMemoryObjectFromStruct;
using ::mlir::triton::gpu::getTotalElemsPerThread;
using ::mlir::triton::gpu::SharedEncodingAttr;

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned numArguments = op.getNumOperands();

    // Currently, Triton kernel function always return nothing.
    // TODO(Superjomn) add support for non-inline device function
    if (numArguments > 0) {
      return rewriter.notifyMatchFailure(
          op, "Only kernel function with nothing returned is supported.");
    }

    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    assert(false);
    return success();
  }
};
} // namespace

namespace AMD {
void populateTritonGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns, int numWarps,
                                     ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                     ModuleAllocation &moduleAllocation,
                                     PatternBenefit benefit) {
  patterns.add<ReturnOpConversion>(typeConverter, benefit);

  mlir::triton::populateMemoryOpToLLVMPattern(typeConverter, patterns, benefit);
  mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, patterns,
                                                 benefit);
  mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns, benefit);
}

} // namespace AMD
