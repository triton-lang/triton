#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUMMALOWERINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

class SyncMMALowering : public OpInterfaceRewritePattern<MMAv5OpInterface> {
public:
  using OpInterfaceRewritePattern<MMAv5OpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(MMAv5OpInterface op,
                                PatternRewriter &rewriter) const override {
    // If the op doesn't have synchronous semantic skip the pattern.
    if (op.isAsync())
      return failure();
    MLIRContext *ctx = op.getContext();
    Location loc = op.getLoc();
    Attribute sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
    auto barrierCTALayout = ttg::CTALayoutAttr::get(
        /*context=*/ctx, /*CTAsPerCGA=*/{1},
        /*CTASplitNum=*/{1}, /*CTAOrder=*/{0});
    auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
        ctx, 1, 1, 1, {0}, barrierCTALayout);
    ttg::MemDescType barrierMemDescType =
        ttg::MemDescType::get({1}, rewriter.getI64Type(), barrierEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);
    Value barrierAlloc =
        rewriter.create<ttg::LocalAllocOp>(loc, barrierMemDescType, Value());
    rewriter.create<InitBarrierOp>(loc, barrierAlloc, 1);
    op.addCompletionBarrier(barrierAlloc,
                            rewriter.create<arith::ConstantIntOp>(loc, 1, 1));
    op.setIsAsync(true);

    rewriter.setInsertionPointAfter(op);
    Value phase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    rewriter.create<WaitBarrierOp>(loc, barrierAlloc, phase, op.getPredicate());
    rewriter.create<InvalBarrierOp>(loc, barrierAlloc);
    return success();
  }
};

struct TCGen5MMAScaleSharedToTmemConversion
    : public OpRewritePattern<TCGen5MMAScaledOp> {
  using OpRewritePattern<TCGen5MMAScaledOp>::OpRewritePattern;

  // Create a tmem_copy of scales from shared memory to tmem. `rows` is the M or
  // N of the MMA operation (for LHS or RHS respectively).
  bool lowerScaleToTmem(OpOperand &operand, PatternRewriter &rewriter,
                        int rows) const {
    Location loc = operand.getOwner()->getLoc();
    MLIRContext *context = operand.getOwner()->getContext();
    Attribute tensorMemorySpace = TensorMemorySpaceAttr::get(context);
    auto oldType = cast<ttg::MemDescType>(operand.get().getType());
    auto numElems = product(oldType.getShape());
    Type elType = oldType.getElementType();
    ttg::CTALayoutAttr CTALayout = ttg::getCTALayout(oldType.getEncoding());
    ArrayRef<unsigned> CTASplitNum = CTALayout.getCTASplitNum();
    // Distribute the scales across the rows of the MMA operation.
    SmallVector<int64_t> shape = {rows, numElems / rows};
    Attribute scaleEncoding = TensorMemoryScalesEncodingAttr::get(
        context, CTASplitNum[0], CTASplitNum[1]);
    Type scaleAType =
        ttg::MemDescType::get(shape, elType, scaleEncoding, tensorMemorySpace,
                              /*mutableMemory=*/true);
    auto tmemAlloc = rewriter.create<TMEMAllocOp>(loc, scaleAType, Value());
    rewriter.create<TMEMCopyOp>(loc, operand.get(), tmemAlloc,
                                /*barrier*/ Value());
    operand.set(tmemAlloc);
    return true;
  }

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    MLIRContext *context = op->getContext();
    auto aScaleType = op.getAScale().getType();
    auto bScaleType = op.getBScale().getType();
    int blockM = op.getBlockM();
    int blockN = op.getBlockN();
    bool anyChanged = false;
    if (isa<ttg::SharedMemorySpaceAttr>(aScaleType.getMemorySpace())) {
      anyChanged = lowerScaleToTmem(op.getAScaleMutable(), rewriter, blockM);
    }
    if (isa<ttg::SharedMemorySpaceAttr>(bScaleType.getMemorySpace())) {
      anyChanged = lowerScaleToTmem(op.getBScaleMutable(), rewriter, blockN);
    }
    return LogicalResult::success(anyChanged);
  }
};

std::pair<SmallVector<TCGen5CommitOp>, SmallVector<Value>>
collectCommitOpsAfter(MMAv5OpInterface mmaOp) {
  auto isConstTrue = [](Value v) {
    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<BoolAttr>(constOp.getValueAttr())) {
        return attr.getValue();
      }
    }
    return false;
  };

  SmallVector<TCGen5CommitOp> commitOps;
  SmallVector<Value> commitPredicates;
  auto mmaPred = mmaOp.getPredicate();
  Operation *nextOp = mmaOp->getNextNode();

  while (nextOp) {
    if (auto commit = dyn_cast<TCGen5CommitOp>(nextOp)) {
      // If the mma predicate is true, or mma and commit ops use the same
      // predicate, it is safe to merge them
      if (isConstTrue(mmaPred) || mmaPred == commit.getPred()) {
        commitOps.push_back(commit);
        commitPredicates.push_back(commit.getPred());
      }
    } else if (!isPure(nextOp)) {
      // Only move commits across pure ops. We also bail here when encountering
      // another MMAv5 op.
      break;
    }
    nextOp = nextOp->getNextNode();
  }

  return {commitOps, commitPredicates};
}

// Return false if defining ops cannot be moved above the target op
bool moveDefiningOpsBefore(Value val, Operation *target) {
  SetVector<Operation *> toMove;

  std::function<bool(Value)> collectOpsToMove = [&](Value val) {
    if (auto defOp = val.getDefiningOp()) {
      if (defOp->getBlock() == target->getBlock() &&
          target->isBeforeInBlock(defOp)) {
        if (!isPure(defOp)) {
          // This defOp needs to move above the target op, but it is unsafe due
          // to impurity.
          return false;
        }
        for (Value operand : defOp->getOperands()) {
          if (!collectOpsToMove(operand)) {
            return false;
          }
        }
        toMove.insert(defOp);
      }
    }
    return true;
  };

  if (!collectOpsToMove(val)) {
    return false;
  }

  for (Operation *op : toMove) {
    op->moveBefore(target);
  }

  return true;
}

class MergeCommitIntoMMA : public OpInterfaceRewritePattern<MMAv5OpInterface> {
public:
  using OpInterfaceRewritePattern<MMAv5OpInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(MMAv5OpInterface op,
                                PatternRewriter &rewriter) const override {
    auto [commitOps, predicates] = collectCommitOpsAfter(op);
    if (commitOps.size() == 0) {
      return llvm::failure();
    }
    for (auto [commit, pred] : llvm::zip(commitOps, predicates)) {
      if (!pred) {
        pred = rewriter.create<arith::ConstantIntOp>(op.getLoc(), true, 1);
      }
      if (!moveDefiningOpsBefore(commit.getBarrier(), op) ||
          !moveDefiningOpsBefore(pred, op)) {
        // Give up merging a commit if its defining ops cannot be moved above
        // the mma op.
        continue;
      }
      op.addCompletionBarrier(commit.getBarrier(), pred);
      rewriter.eraseOp(commit);
    }
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUMMALoweringPass
    : public impl::TritonNvidiaGPUMMALoweringPassBase<
          TritonNvidiaGPUMMALoweringPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);
    patterns.add<SyncMMALowering, TCGen5MMAScaleSharedToTmemConversion,
                 MergeCommitIntoMMA>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
