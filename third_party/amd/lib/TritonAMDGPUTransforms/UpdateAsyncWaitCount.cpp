
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-update-async-wait-count"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace ttg = triton::gpu;

unsigned countLoadInstructions(MLIRContext *context, RankedTensorType srcTy,
                               ttg::MemDescType dstTy) {
  auto shape = srcTy.getShape();
  LinearLayout srcLayout =
      triton::gpu::toLinearLayout(shape, srcTy.getEncoding());
  LinearLayout sharedLayout =
      triton::gpu::toLinearLayout(shape, dstTy.getEncoding());
  LinearLayout srcToSharedLayout = srcLayout.invertAndCompose(sharedLayout);

  // On GFX9 we cannot split direct to lds loads into multiple ones because we
  // need coalesced writes. So we can divide the number of registers by the
  // contiguity to get the number of load instructions.
  unsigned contig = srcToSharedLayout.getNumConsecutiveInOut();
  unsigned numberOfRegisters =
      srcToSharedLayout.getInDimSize(StringAttr::get(context, "register"));

  unsigned loadInstructionCount = numberOfRegisters / contig;

  return loadInstructionCount;
}

unsigned countLoadInstructions(MLIRContext *context,
                               ttg::AsyncCopyGlobalToLocalOp copyOp) {
  return countLoadInstructions(context, copyOp.getSrc().getType(),
                               copyOp.getResult().getType());
}

unsigned countLoadInstructions(MLIRContext *context,
                               amdgpu::BufferLoadToLocalOp copyOp) {
  RankedTensorType srcTy = cast<RankedTensorType>(
      LLVM::AMD::getPointerTypeWithShape(copyOp.getPtr(), copyOp.getOffsets()));
  return countLoadInstructions(context, srcTy, copyOp.getDest().getType());
}

// LLVM cannot infer the dependency between direct to lds (async) loads and the
// local reads between warps in a workgroup. As a workaround we can count the
// instructions of load and store instruction and emit the correct waitcnt
// ourselfs. It is important to never overestimate or else we will see
// correctness issues.
struct UpdateAsyncWaitCount : public OpRewritePattern<ttg::AsyncWaitOp> {
  UpdateAsyncWaitCount(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(ttg::AsyncWaitOp waitOp,
                                PatternRewriter &rewriter) const override {
    int minCommitNumber = INT_MAX;

    if (waitOp->getNumOperands() < 1) {
      return failure();
    }

    int waitCnt = std::numeric_limits<int>::max();

    for (auto operand : waitOp.getOperands()) {
      // If the value resides in a region other than the region of the wait op,
      // then the wait op must be in some nested region. Measure the number of
      // commits between the definition value and the parent op.
      // TODO: We could measure commits in nested regions along the path if
      // necessary.
      Operation *waitOpPtr = waitOp;
      while (waitOp->getParentRegion() != operand.getParentRegion())
        waitOpPtr = waitOp->getParentOp();
      waitCnt = std::min(waitCnt, findMinCountInDefChain(operand, waitOpPtr));
    }

    if (waitCnt == std::numeric_limits<int>::max() ||
        waitOp.getNum() == waitCnt)
      return failure();

    rewriter.modifyOpInPlace(waitOp, [&]() { waitOp.setNum(waitCnt); });
    return success();
  }

  // DFS the def chain of val from sinkOp and call countOp on all operation
  // ranges spanned by the def chain. Returns the minimum count found in all def
  // chain paths
  // TODO: merge with minNumInterleavedCommitOps
  int findMinCountInDefChain(
      Value val, Operation *sinkOp, int pathSum = 0,
      int foundMin = std::numeric_limits<int>::max()) const {
    if (Operation *defOp = val.getDefiningOp()) {
      pathSum += countVMCntInstructionBetween(defOp->getNextNode(), sinkOp);
      foundMin = std::min(foundMin, pathSum);
      return foundMin;
    }
    if (auto arg = mlir::dyn_cast<BlockArgument>(val)) {
      Block *block = arg.getOwner();
      auto forOp = dyn_cast<scf::ForOp>(block->getParentOp());

      // Failed to track, return 0 conservatively.
      if (!forOp)
        return 0;

      Operation *firstForInst = &*forOp.getBody()->begin();
      int insertsBetween = countVMCntInstructionBetween(firstForInst, sinkOp);
      pathSum += insertsBetween;
      if (pathSum >= foundMin)
        return foundMin;

      // get the value assigned to the argument coming from outside the loop
      Value incomingVal = forOp.getInitArgs()[arg.getArgNumber() - 1];
      int min1 = findMinCountInDefChain(incomingVal, forOp, pathSum, foundMin);

      // get the value assigned to the argument coming from the previous
      // iteration
      Operation *yieldOp = block->getTerminator();
      Value prevVal = yieldOp->getOperand(arg.getArgNumber() - 1);
      int min2 = findMinCountInDefChain(prevVal, yieldOp, pathSum, foundMin);
      return std::min(std::min(min1, min2), foundMin);
    }
    // Failed to track, return 0 conservatively.
    return 0;
  }

  int countVMCntInstructionBetween(Operation *op1, Operation *op2) const {
    auto *ctx = getContext();
    int count = 0;
    for (auto op = op1; op != op2; op = op->getNextNode()) {
      if (isa<ttg::AsyncCommitGroupOp>(op)) {
        Value token = op->getOperand(0);
        auto defToken = token.getDefiningOp();
        if (defToken) {
          count += llvm::TypeSwitch<Operation *, unsigned>(defToken)
                       .Case([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
                         return countLoadInstructions(ctx, copyOp);
                       })
                       .Case([&](amdgpu::BufferLoadToLocalOp copyOp) {
                         return countLoadInstructions(ctx, copyOp);
                       })
                       .Case([&](triton::StoreOp loadOp) {
                         // We do not control the vectorSize for global
                         // stores so just return 0 which will always be
                         // correct
                         LDBG("Global store between async waits. We will "
                              "conservatively set the vmcnt which might "
                              "impact performance.");
                         return 0;
                       })
                       .Case([&](triton::LoadOp loadOp) {
                         // We do not control the vectorSize for global
                         // loads so just return 0 which will always be
                         // correct
                         LDBG("Global load between async waits. We will "
                              "conservatively set the vmcnt which might "
                              "impact performance.");
                         return 0;
                       })
                       .Default([&](auto) { return 0; });
        }
      }
    }
    return count;
  };
};

class TritonAMDGPUUpdateAsyncWaitCountPass
    : public TritonAMDGPUUpdateAsyncWaitCountBase<
          TritonAMDGPUUpdateAsyncWaitCountPass> {
public:
  TritonAMDGPUUpdateAsyncWaitCountPass(StringRef archGenName) {
    this->archGenerationName = archGenName.str();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    triton::AMD::TargetInfo targetInfo(archGenerationName);
    switch (targetInfo.getISAFamily()) {
    case triton::AMD::ISAFamily::CDNA1:
    case triton::AMD::ISAFamily::CDNA2:
    case triton::AMD::ISAFamily::CDNA3:
    case triton::AMD::ISAFamily::CDNA4: {
      break;
    }
    default:
      return;
    }

    mlir::RewritePatternSet patterns(context);

    // Precompute the contiguity of all AsyncCopy ops based on the src and
    // mask contiguity/alignment to avoid rebuilding ModuleAxisInfoAnalysis
    // after every IR change.
    patterns.add<UpdateAsyncWaitCount>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUUpdateAsyncWaitCountPass(std::string archGenName) {
  return std::make_unique<TritonAMDGPUUpdateAsyncWaitCountPass>(archGenName);
}
