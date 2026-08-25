#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/BufferRegion.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUMMALOWERINGPASS
#define GEN_PASS_DEF_TRITONNVIDIAGPUNORMALIZEMMAKPASS
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
    auto numCTAs = gpu::lookupNumCTAs(op);
    auto barrierCGALayout = ttg::CGAEncodingAttr::get1DLayout(ctx, numCTAs);
    auto barrierEncoding = ttg::SwizzledSharedEncodingAttr::get(
        ctx, 1, 1, 1, {0}, barrierCGALayout);
    ttg::MemDescType barrierMemDescType =
        ttg::MemDescType::get({numCTAs}, rewriter.getI64Type(), barrierEncoding,
                              sharedMemorySpace, /*mutableMemory=*/true);
    Value barrierAlloc =
        ttg::LocalAllocOp::create(rewriter, loc, barrierMemDescType, Value());
    InitBarrierOp::create(rewriter, loc, barrierAlloc, 1);
    op.addCompletionBarrier(barrierAlloc,
                            arith::ConstantIntOp::create(rewriter, loc, 1, 1));
    op.setIsAsync(true);

    rewriter.setInsertionPointAfter(op);
    Value phase = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    WaitBarrierOp::create(rewriter, loc, barrierAlloc, phase,
                          op.getPredicate());
    InvalBarrierOp::create(rewriter, loc, barrierAlloc);
    return success();
  }
};

struct TCGen5MMAScaleSharedToTmemConversion
    : public OpRewritePattern<TCGen5MMAScaledOp> {
  using OpRewritePattern<TCGen5MMAScaledOp>::OpRewritePattern;

  // Create a tmem_copy of scales from shared memory to tmem. `rows` is the M or
  // N of the MMA operation (for LHS or RHS respectively).
  bool lowerScaleToTmem(OpOperand &operand, PatternRewriter &rewriter, int rows,
                        TensorMemoryScalesBlockRepOrder blockRepOrder) const {
    Location loc = operand.getOwner()->getLoc();
    MLIRContext *context = operand.getOwner()->getContext();
    Attribute tensorMemorySpace = TensorMemorySpaceAttr::get(context);
    auto oldType = cast<ttg::MemDescType>(operand.get().getType());
    auto numElems = product(oldType.getShape());
    Type elType = oldType.getElementType();
    ttg::CGAEncodingAttr CGALayout = ttg::getCGALayout(oldType.getEncoding());
    // Distribute the scales across the rows of the MMA operation.
    SmallVector<int64_t> shape = {rows, numElems / rows};
    Attribute scaleEncoding =
        TensorMemoryScalesEncodingAttr::get(context, CGALayout, blockRepOrder);
    Type scaleAType =
        ttg::MemDescType::get(shape, elType, scaleEncoding, tensorMemorySpace,
                              /*mutableMemory=*/true);
    auto tmemAlloc = TMEMAllocOp::create(rewriter, loc, scaleAType, Value());
    TMEMCopyOp::create(rewriter, loc, operand.get(), tmemAlloc);
    operand.set(tmemAlloc);
    return true;
  }

  LogicalResult matchAndRewrite(TCGen5MMAScaledOp op,
                                PatternRewriter &rewriter) const override {
    auto aScaleType = op.getAScale().getType();
    auto bScaleType = op.getBScale().getType();
    if ((isa<ttg::SharedMemorySpaceAttr>(aScaleType.getMemorySpace()) &&
         aScaleType.getShape() != aScaleType.getAllocShape()) ||
        (isa<ttg::SharedMemorySpaceAttr>(bScaleType.getMemorySpace()) &&
         bScaleType.getShape() != bScaleType.getAllocShape())) {
      op.emitError("subviews NYI");
      return failure();
    }
    int blockM = op.getBlockM();
    int blockN = op.getBlockN();
    auto aScaleBlockRepOrder = getTensorMemoryScalesBlockRepOrder(
        op, /*isA=*/true, op.getAType(), op.getBType(),
        aScaleType.getElementType(), bScaleType.getElementType());
    auto bScaleBlockRepOrder = getTensorMemoryScalesBlockRepOrder(
        op, /*isA=*/false, op.getAType(), op.getBType(),
        aScaleType.getElementType(), bScaleType.getElementType());
    bool anyChanged = false;
    if (isa<ttg::SharedMemorySpaceAttr>(aScaleType.getMemorySpace())) {
      anyChanged = lowerScaleToTmem(op.getAScaleMutable(), rewriter, blockM,
                                    aScaleBlockRepOrder) ||
                   anyChanged;
    }
    if (isa<ttg::SharedMemorySpaceAttr>(bScaleType.getMemorySpace())) {
      anyChanged = lowerScaleToTmem(op.getBScaleMutable(), rewriter, blockN,
                                    bScaleBlockRepOrder) ||
                   anyChanged;
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
  SmallVector<Value> mmaDescs = mmaOp.getCompletionDescs();

  while (nextOp) {
    if (auto commit = dyn_cast<TCGen5CommitOp>(nextOp)) {
      // If the mma predicate is true, or mma and commit ops use the same
      // predicate, it is safe to merge them. Otherwise, keep commit order by
      // not merging later commits across this one.
      if (!isConstTrue(mmaPred) && mmaPred != commit.getPred())
        break;
      if (!llvm::equal(mmaDescs, commit.getDescs()))
        break;
      commitOps.push_back(commit);
      commitPredicates.push_back(commit.getPred());
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
    if (commitOps.empty()) {
      return llvm::failure();
    }
    for (auto [commit, pred] : llvm::zip(commitOps, predicates)) {
      if (!pred) {
        pred = arith::ConstantIntOp::create(rewriter, op.getLoc(), true, 1);
      }
      Value barrier = commit.getBarrier();
      if (!moveDefiningOpsBefore(barrier, op) ||
          !moveDefiningOpsBefore(pred, op)) {
        // Give up merging a commit if its defining ops cannot be moved above
        // the mma op.
        break;
      }
      op.addCompletionBarrier(barrier, pred);
      rewriter.eraseOp(commit);
    }
    return success();
  }
};

} // anonymous namespace

class TritonNvidiaGPUNormalizeMMAKPass
    : public impl::TritonNvidiaGPUNormalizeMMAKPassBase<
          TritonNvidiaGPUNormalizeMMAKPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<TCGen5MMAScaledOp> mmas;
    module.walk([&](TCGen5MMAScaledOp op) {
      if (!op.getK96Instructions(/*allowUnnormalized=*/true).empty() ||
          op.getKRangeAttr())
        mmas.push_back(op);
    });
    if (mmas.empty())
      return;
    auto solver = createDataFlowSolver();
    auto *regions =
        solver->load<BufferRegionAnalysis>(/*relativeToAllocation=*/true);
    if (failed(solver->initializeAndRun(module))) {
      signalPassFailure();
      return;
    }
    for (auto op : mmas) {
      bool automatic = !op.getInstructionK() && !op.getKRangeAttr();
      if (automatic)
        op.removeKBaseOffsetsAttr();
      auto tiles = op.getK96Instructions(/*allowUnnormalized=*/true);
      bool hasK96 = !tiles.empty();
      if (tiles.empty())
        for (int k = op.getKStart(); k < op.getKStart() + op.getBlockK();
             k += op.getInstructionK().value_or(64))
          tiles.push_back({k, int(op.getInstructionK().value_or(64))});
      auto invalid = [&](StringRef message) {
        if (!automatic)
          op.emitError(message);
        return failure();
      };
      SmallVector<int32_t> bases(4, 0);
      SetVector<ttg::LocalAllocOp> allocations;
      auto getBase = [&](Value value, unsigned index) -> LogicalResult {
        if (!value)
          return success();
        const auto &info = regions->getRegionInfo(value);
        if (info.kind != RegionInfo::Kind::Exact || info.views.empty())
          return invalid(
              "cannot prove the physical K origin of the MMA operand");
        std::optional<int32_t> phase;
        for (const auto &view : info.views) {
          auto alloc = dyn_cast<ttg::LocalAllocOp>(
              regions->getAllocation(view.allocationFrame));
          if (!alloc)
            return invalid("selected MMA operands must originate in "
                           "shared-memory allocations");
          // The hardware absolute-stride form requires matrix base offset zero.
          // Reinterprets retain allocation ownership; strengthen that
          // allocation's alignment instead of manufacturing a new base address.
          allocations.insert(alloc);
          uint32_t offset = view.region.baseOffset;
          if ((offset & 15) || (hasK96 && (offset & 0x380)))
            return invalid("K96 requires a 16-byte aligned origin and zero "
                           "matrix base offset");
          int32_t current = (offset & 127) * 2;
          if (phase && *phase != current)
            return invalid(
                "MMA ring views must have a consistent physical K alignment");
          phase = current;
        }
        bases[index] = *phase;
        return success();
      };
      int end = op.getKStart() + op.getBlockK();
      Value aNext =
          end > op.getA().getType().getShape()[1] * 2 ? op.getANext() : Value();
      Value bNext =
          end > op.getB().getType().getShape()[0] * 2 ? op.getBNext() : Value();
      if (failed(getBase(op.getA(), 0)) || failed(getBase(op.getB(), 1)) ||
          failed(getBase(aNext, 2)) || failed(getBase(bNext, 3))) {
        if (automatic)
          continue;
        signalPassFailure();
        return;
      }
      auto verifySource = [&](Value first, Value next,
                              bool lhs) -> LogicalResult {
        int capacity =
            cast<ttg::MemDescType>(first.getType()).getShape()[lhs ? 1 : 0] * 2;
        int firstBase = bases[lhs ? 0 : 1];
        int nextBase = bases[lhs ? 2 : 3];
        for (auto tile : tiles) {
          bool inNext = tile.k >= capacity;
          int k = inNext ? tile.k - capacity : tile.k;
          int base = inNext ? nextBase : firstBase;
          int remaining = 256 - (base + k) % 256;
          bool crossesView = !inNext && tile.k + tile.width > capacity;
          if (crossesView) {
            if (!next || tile.width != 96 || capacity - tile.k != remaining)
              return invalid("continuation boundary must coincide with the "
                             "physical 128-byte K boundary");
            if (nextBase + tile.width - remaining > 256)
              return invalid("continuation chunk crosses a second physical "
                             "128-byte K boundary");
          } else if (tile.width > remaining && tile.width != 96) {
            return invalid(
                "only K96 supports crossing a physical 128-byte K boundary");
          }
        }
        return success();
      };
      if (failed(verifySource(op.getA(), op.getANext(), true)) ||
          failed(verifySource(op.getB(), op.getBNext(), false))) {
        if (automatic)
          continue;
        signalPassFailure();
        return;
      }
      if (hasK96)
        for (auto alloc : allocations)
          if (alloc.getAlignmentOrDefault() < 1024)
            alloc.setAlignment(1024);
      op.setKBaseOffsetsAttr(DenseI32ArrayAttr::get(&getContext(), bases));
    }
  }
};

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
