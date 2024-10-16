#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#include <deque>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {
bool verifyNonNegativeByAssumption(Value expr,
                                   const DenseSet<Value> &assumptions) {
  for (Value assume : assumptions) {
    LDBG("Assumption:" << assume);
    if (auto cmpOp = assume.getDefiningOp<arith::CmpIOp>()) {
      bool isGreaterThan = (cmpOp.getPredicate() == arith::CmpIPredicate::sge ||
                            cmpOp.getPredicate() == arith::CmpIPredicate::sgt);
      APInt cst;
      if (isGreaterThan && (cmpOp.getLhs() == expr) &&
          matchPattern(cmpOp.getRhs(), m_ConstantInt(&cst))) {
        return cst.isNonNegative();
      }
    }
  }
  return false;
}

bool verifyNonNegativeExpr(Value expr, const DenseSet<Value> &assumptions) {

  // Check if the expression is contained in any assumption
  if (verifyNonNegativeByAssumption(expr, assumptions)) {
    LDBG("Non negative by assumption");
    return true;
  }

  // Recurse if the operation is defined
  Operation *op = expr.getDefiningOp();
  if (!op)
    return false;

  bool nonNegative =
      llvm::TypeSwitch<Operation *, bool>(expr.getDefiningOp())
          .Case<triton::BroadcastOp>([&](auto broadcastOp) {
            return verifyNonNegativeExpr(broadcastOp.getSrc(), assumptions);
          })
          .Case<triton::ExpandDimsOp>([&](auto expandOp) {
            return verifyNonNegativeExpr(expandOp.getSrc(), assumptions);
          })
          .Case<triton::SplatOp>([&](auto splatOp) {
            return verifyNonNegativeExpr(splatOp.getSrc(), assumptions);
          })
          .Case<triton::MakeRangeOp>([&](auto makeRangeOp) {
            return makeRangeOp.getStart() >= 0 && makeRangeOp.getEnd() >= 0;
          })
          .Case<arith::ConstantIntOp>(
              [&](auto constIntOp) { return constIntOp.value() >= 0; })
          .Case<arith::ConstantOp>([&](arith::ConstantOp constOp) {
            Value val = constOp.getResult();
            DenseIntElementsAttr constVal;
            if (matchPattern(val, m_Constant(&constVal)) && constVal.isSplat())
              return constVal.getSplatValue<APInt>().isNonNegative();
            return false;
          })
          .Case<triton::GetProgramIdOp>([&](auto pidOp) { return true; })
          .Case<arith::MaxSIOp>([&](auto maxOp) {
            // max(a,b) >= 0 iff a>=0 || b>=0
            bool nnLhs = verifyNonNegativeExpr(maxOp.getLhs(), assumptions);
            bool nnRhs = verifyNonNegativeExpr(maxOp.getRhs(), assumptions);
            return nnLhs || nnRhs;
          })
          .Case<arith::RemSIOp>([&](auto remsiOp) {
            // a % b >= 0 iff a>=0
            return verifyNonNegativeExpr(remsiOp.getLhs(), assumptions);
          })
          .Case<arith::TruncIOp, arith::ExtSIOp>([&](Operation *unaryOp) {
            // a = OP b >= 0 iff b >= 0
            return verifyNonNegativeExpr(unaryOp->getOperand(0), assumptions);
          })
          .Case<arith::AddIOp, arith::MinSIOp, arith::MulIOp, arith::DivSIOp>(
              // Generally speaking, a OP b >= 0  iff  a >= 0 && b >= 0 when
              // OP != sub
              [&](Operation *binOp) {
                bool nnLhs =
                    verifyNonNegativeExpr(binOp->getOperand(0), assumptions);
                bool nnRhs =
                    verifyNonNegativeExpr(binOp->getOperand(1), assumptions);
                return nnLhs && nnRhs;
              })
          .Default([&](Operation *op) {
            // Conservatively assume that the expression is negative
            return false;
          });
  return nonNegative;
}

// Quick analysis on the Triton IR to decide if we can safely use
// buffer operations
bool canUseBufferOps(Value ptr, const DenseSet<Value> &assumptions) {
  // 1. Check if the pointer is uniform: i.e., if it comes from a uniform
  // pointer(splatted) and non-uniform offset addition
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  bool useBufferOps =
      addPtrOp && isa<triton::SplatOp>(addPtrOp.getPtr().getDefiningOp());
  LDBG("Pattern matched: " << useBufferOps);

  // 2. Check if the offset is a 32-bit tensor
  Value offset = addPtrOp.getOffset();
  useBufferOps =
      useBufferOps &&
      (cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth() == 32);
  LDBG("32 bit offset: " << useBufferOps);

  // 3. Check if the offset is non-negative
  useBufferOps = useBufferOps && verifyNonNegativeExpr(offset, assumptions);
  LDBG("Non-negative: " << useBufferOps);
  return useBufferOps;
}
} // namespace

struct ConvertTritonLoadToBufferLoad
    : public mlir::OpRewritePattern<triton::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(mlir::MLIRContext *context,
                                DenseSet<Value> &assumptions)
      : mlir::OpRewritePattern<triton::LoadOp>(context),
        assumptions(assumptions) {}

  mlir::LogicalResult
  matchAndRewrite(triton::LoadOp op, PatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();

    if (op.getCache() != triton::CacheModifier::NONE)
      return failure();

    if (canUseBufferOps(ptr, assumptions)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferLoadOp>(
          op, op.getType(), basePtr, tensorOffset, op.getMask(), op.getOther());
      return success();
    }
    return failure();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(mlir::MLIRContext *context,
                                  DenseSet<Value> &assumptions)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();

    if (op.getCache() != triton::CacheModifier::NONE)
      return failure();

    if (canUseBufferOps(ptr, assumptions)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value tensorOffset = addPtrOp.getOffset();
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, op.getMask());
      return success();
    }
    return failure();
  }

private:
  // Assumptions collected through the function
  DenseSet<Value> assumptions;
};

class TritonAMDGPUConvertToBufferOpsPass
    : public TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {

public:
  TritonAMDGPUConvertToBufferOpsPass() = default;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    triton::FuncOp f = getOperation();
    // Collect assumptions in the function
    DenseSet<Value> assumptions;
    f.walk([&](LLVM::AssumeOp op) { assumptions.insert(op->getOperand(0)); });
    LDBG("Number of assumptions found: " << assumptions.size());

    patterns.add<ConvertTritonLoadToBufferLoad>(context, assumptions);
    patterns.add<ConvertTritonStoreToBufferStore>(context, assumptions);
    if (applyPatternsAndFoldGreedily(f, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUConvertToBufferOpsPass() {
  return std::make_unique<TritonAMDGPUConvertToBufferOpsPass>();
}
