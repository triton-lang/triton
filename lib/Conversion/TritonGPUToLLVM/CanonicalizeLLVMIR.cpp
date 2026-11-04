#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_CANONICALIZELLVMIR
#include "triton/Conversion/TritonGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
class FoldAbsIntoReduxPattern : public OpRewritePattern<NVVM::ReduxOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(NVVM::ReduxOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getKind() != NVVM::ReductionKind::FMAX &&
        op.getKind() != NVVM::ReductionKind::FMIN)
      return failure();
    auto abs = op.getVal().getDefiningOp<LLVM::FAbsOp>();
    if (!abs)
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      op.getValMutable().assign(abs.getOperand());
      op.setAbs(true);
    });
    return success();
  }
};

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

class ElideFullClusterRankMaskPattern : public OpRewritePattern<LLVM::AndOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::AndOp op,
                                PatternRewriter &rewriter) const override {
    APInt mask;
    Value rank = op.getLhs();
    if (!matchPattern(op.getRhs(), m_ConstantInt(&mask))) {
      if (!matchPattern(op.getLhs(), m_ConstantInt(&mask)))
        return failure();
      rank = op.getRhs();
    }

    if (!rank.getDefiningOp<triton::nvgpu::ClusterCTAIdOp>())
      return failure();

    unsigned numCTAs = triton::gpu::lookupNumCTAs(op);
    if (mask.countr_one() < llvm::Log2_32_Ceil(numCTAs))
      return failure();

    rewriter.replaceOp(op, rank);
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
    patterns.add<SelectConstantConditionPattern,
                 ElideFullClusterRankMaskPattern, FoldAbsIntoReduxPattern>(
        &getContext());

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
