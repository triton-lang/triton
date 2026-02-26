#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_CANONICALIZELLVMIR
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
class SelectConstantConditionPattern : public OpRewritePattern<LLVM::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::SelectOp op,
                                PatternRewriter &b) const override {
    BoolAttr cond;
    if (!matchPattern(op.getCondition(), m_Constant(&cond)))
      return failure();
    Value val = cond.getValue() ? op.getTrueValue() : op.getFalseValue();
    b.replaceOp(op, ValueRange{val});
    return success();
  }
};
} // namespace

namespace {
struct CanonicalizeLLVMIR
    : public mlir::triton::gpu::impl::CanonicalizeLLVMIRBase<
          CanonicalizeLLVMIR> {
  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<SelectConstantConditionPattern>(&getContext());

    getContext()
        .getLoadedDialect<LLVM::LLVMDialect>()
        ->getCanonicalizationPatterns(patterns);
    for (mlir::RegisteredOperationName op :
         getContext().getRegisteredOperationsByDialect(
             LLVM::LLVMDialect::getDialectNamespace()))
      op.getCanonicalizationPatterns(patterns, &getContext());

    (void)applyPatternsGreedily(func, std::move(patterns));
  }
};
} // namespace
