#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include <memory>

// using namespace mlir;

namespace {
// dot(a, b, 0) + c => dot(a, b, c)
class CombineDotOp : public mlir::RewritePattern {
public:
  CombineDotOp(mlir::MLIRContext *context)
    : mlir::RewritePattern(mlir::RewritePattern::MatchAnyOpTypeTag(), 1, context) {}
  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (llvm::isa<mlir::arith::AddIOp, mlir::arith::AddFOp>(op)) {
      if (isCandidate(op->getOperand(0)).succeeded()) {
        auto dotOp = op->getOperand(0).getDefiningOp<mlir::triton::DotOp>();
        rewriter.replaceOpWithNewOp<mlir::triton::DotOp>(
          op, dotOp->getResultTypes().front(), dotOp.a(),
          dotOp.b(), op->getOperand(1), dotOp.allowTF32());
        return mlir::success();
      } else if (isCandidate(op->getOperand(1)).succeeded()) {
        auto dotOp = op->getOperand(1).getDefiningOp<mlir::triton::DotOp>();
        rewriter.replaceOpWithNewOp<mlir::triton::DotOp>(
          op, dotOp->getResultTypes().front(), dotOp.a(),
          dotOp.b(), op->getOperand(0), dotOp.allowTF32());
        return mlir::success();
      }
    }
    return mlir::failure();
  }

private:
  // Is this value a dot and has 0 as `c`.
  mlir::LogicalResult isCandidate(mlir::Value val) const {
    if (auto dot = val.getDefiningOp<mlir::triton::DotOp>()) {
      if (isZero(dot.c()))
        return mlir::success();
    }
    return mlir::failure();
  }

  bool isZero(mlir::Value val) const {
    if (mlir::matchPattern(val, mlir::m_Zero()) ||
        mlir::matchPattern(val, mlir::m_AnyZeroFloat()))
      return true;
    // broadcast(constant_0)
    if (auto bc = val.getDefiningOp<mlir::triton::BroadcastOp>()) {
      if (mlir::matchPattern(bc.src(), mlir::m_Zero()) || 
          mlir::matchPattern(bc.src(), mlir::m_AnyZeroFloat()))
        return true;
    }
    return false;
  }
};

// gep(gep(%ptr, %idx0), %idx1) => gep(%ptr, AddI(%idx0, %idx1))
//   Note: leave (sub %c0, %c0) canceling to ArithmeticDialect
//         (ref: ArithmeticCanonicalization.td)
class CombineGEPOp : public mlir::RewritePattern {
public:
  CombineGEPOp(mlir::MLIRContext *context)
    : mlir::RewritePattern(mlir::RewritePattern::MatchAnyOpTypeTag(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (llvm::isa<mlir::triton::GEPOp>(op)) {
      if (auto gep2 = op->getOperand(0).getDefiningOp<mlir::triton::GEPOp>()) {
        auto loc = op->getLoc();
        mlir::Value newIdx = rewriter.create<mlir::arith::AddIOp>(
          loc, op->getOperand(1), gep2->getOperand(1));
        rewriter.replaceOpWithNewOp<mlir::triton::GEPOp>(
          op, op->getResultTypes().front(), gep2->getOperand(0), newIdx
        );
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

// select(cond, load(ptrs, broadcast(cond), ???), other)
//   => load(ptrs, broadcast(cond), other)
class CombineSelectMaskedLoadOp : public mlir::RewritePattern {
public:
  CombineSelectMaskedLoadOp(mlir::MLIRContext *context)
    : mlir::RewritePattern(mlir::RewritePattern::MatchAnyOpTypeTag(), 1, context) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const override {
    if (llvm::isa<mlir::SelectOp>(op)) {
      if (auto load = op->getOperand(1).getDefiningOp<mlir::triton::LoadOp>()) {
        mlir::Value cond = op->getOperand(0);
        if (auto bc = load.mask().getDefiningOp<mlir::triton::BroadcastOp>()) {
          if (bc.src().getDefiningOp() == cond.getDefiningOp()) {
            rewriter.replaceOpWithNewOp<mlir::triton::LoadOp>(
              op, op->getResultTypes().front(),
              load.ptr(), load.mask(), op->getOperand(2),
              load.cache(), load.evict(), load.isVolatile()
            );
            return mlir::success();
          }
        }
      }
    }
    return mlir::failure();
  }
};
} // anonymous namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

class CombineOpsPass : public TritonCombineOpsBase<CombineOpsPass> {
public:
  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::ModuleOp m = getOperation();

    patterns.add<CombineDotOp>(context);
    patterns.add<CombineSelectMaskedLoadOp>(context);
    patterns.add<CombineGEPOp>(context);
    // patterns.add<CombineReduceOp>(context);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<mlir::Pass> mlir::triton::createCombineOpsPass() {
  return std::make_unique<CombineOpsPass>();
}
