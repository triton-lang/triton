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

/// If we have a pattern of fpext -> redux.{min,max} -> fptrunc -> fpext, try to
/// elide the truncate and re-extend, since we know the value is perfectly
/// representable in the narrower type.
class FoldReduxExtensionPattern : public OpRewritePattern<LLVM::FPExtOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::FPExtOp op,
                                PatternRewriter &rewriter) const override {
    auto trunc = op.getOperand().getDefiningOp<LLVM::FPTruncOp>();
    if (!trunc)
      return failure();
    Type type = trunc.getType();
    if (!type.isF16() && !type.isBF16())
      return failure();
    Value value = trunc.getOperand();
    if (value.getType() != op.getType())
      return failure();

    SmallVector<Value> worklist{value};
    DenseSet<Value> visited;
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (!visited.insert(current).second)
        continue;
      if (auto ext = current.getDefiningOp<LLVM::FPExtOp>()) {
        if (ext.getOperand().getType() != type)
          return failure();
      } else if (auto select = current.getDefiningOp<LLVM::SelectOp>()) {
        worklist.append({select.getTrueValue(), select.getFalseValue()});
      } else if (auto redux = current.getDefiningOp<NVVM::ReduxOp>()) {
        if (redux.getKind() != NVVM::ReductionKind::FMIN &&
            redux.getKind() != NVVM::ReductionKind::FMAX)
          return failure();
        worklist.push_back(redux.getVal());
      } else {
        FloatAttr constant;
        // NaN or infinity constants are fine.
        if (!matchPattern(current, m_Constant(&constant)) ||
            (!constant.getValue().isNaN() && !constant.getValue().isInfinity()))
          return failure();
      }
    }

    // Min/max of extended values is still representable in the source type.
    rewriter.replaceOp(op, value);
    return success();
  }
};

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
    patterns
        .add<SelectConstantConditionPattern, ElideFullClusterRankMaskPattern,
             FoldAbsIntoReduxPattern, FoldReduxExtensionPattern>(&getContext());

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
