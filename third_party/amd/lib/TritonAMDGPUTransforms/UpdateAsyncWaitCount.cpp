#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;

// LLVM cannot infer the dependency between direct to lds (async) loads and the
// local reads between warps in a workgroup. As a workaround we update the
// waitcnt to represent the number of hardware instructions we are interleaving
// with. This allows us to manually emit the waitcnt during lowering.
struct UpdateAsyncWaitCount : public OpRewritePattern<ttg::AsyncWaitOp> {
  UpdateAsyncWaitCount(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(ttg::AsyncWaitOp waitOp,
                                PatternRewriter &rewriter) const override {
    int waitCnt = std::numeric_limits<int>::max();

    // AsyncWait can await multiple tokens so we get the minimum from all
    // tokens
    for (auto token : waitOp.getOperands()) {
      // Traverse def chain from waitOp to the producer of the token and count
      // the minumum number of vmcnt instructions
      auto tokenWaitCnt =
          findMinPathCountInDefChain(token, waitOp, [this](Operation *op) {
            return getNumberOfLoadInstructions(op);
          });
      waitCnt = std::min(waitCnt, tokenWaitCnt);
    }

    if (waitCnt == std::numeric_limits<int>::max() ||
        waitOp.getNum() == waitCnt)
      return failure();

    rewriter.modifyOpInPlace(waitOp, [&]() { waitOp.setNum(waitCnt); });
    return success();
  }

  // To count AsyncLoads we have two options:
  // 1) Count the AsyncLoads directly but we the overall producer of the token
  //    might be an AsyncCommitGroup in which case we would miss the loads of
  //    this CommitGroup.
  // 2) Count based on the Operands of encountered CommitGroups. This ensures we
  //    include the loads of the CommitGroup producing the token. We might miss
  //    AsyncLoads if their commit group is scheduled very late.
  // The second case is implemented because the scheduling will place the
  // CommitGroup right after the AsyncCopy hence we will never miss AsyncCopies
  int getNumberOfLoadInstructions(Operation *op) const {
    if (isa<ttg::AsyncCommitGroupOp>(op)) {
      int count = 0;
      for (auto token : op->getOperands()) {
        auto defOp = token.getDefiningOp();
        if (!defOp)
          continue;
        count += llvm::TypeSwitch<Operation *, int>(defOp)
                     .Case([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
                       return getNumberOfLoadInstructions(
                           getContext(), copyOp.getSrc().getType(),
                           copyOp.getResult().getType());
                     })
                     .Case([&](amdgpu::BufferLoadToLocalOp copyOp) {
                       RankedTensorType srcTy = cast<RankedTensorType>(
                           LLVM::AMD::getPointerTypeWithShape(
                               copyOp.getPtr(), copyOp.getOffsets()));
                       return getNumberOfLoadInstructions(
                           getContext(), srcTy, copyOp.getDest().getType());
                     })
                     .Default([&](auto) { return 0; });
      }
      return count;
    } else if (isa<tt::LoadOp, tt::StoreOp>(op)) {
      op->emitRemark("Global memory instruction between async wait and "
                     "async_loads. This will hinder the interleaving of memory "
                     "operations and might impact performance.");
    }
    return 0;
  }

  // Overload to get the number of loads from direct to lds memory layouts
  int getNumberOfLoadInstructions(MLIRContext *context, RankedTensorType srcTy,
                                  ttg::MemDescType dstTy) const {
    auto shape = srcTy.getShape();
    LinearLayout srcLayout =
        tt::gpu::toLinearLayout(shape, srcTy.getEncoding());
    LinearLayout sharedLayout =
        tt::gpu::toLinearLayout(shape, dstTy.getEncoding());
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

    tt::AMD::TargetInfo targetInfo(archGenerationName);
    switch (targetInfo.getISAFamily()) {
    case tt::AMD::ISAFamily::CDNA1:
    case tt::AMD::ISAFamily::CDNA2:
    case tt::AMD::ISAFamily::CDNA3:
    case tt::AMD::ISAFamily::CDNA4: {
      break;
    }
    default:
      return;
    }

    mlir::RewritePatternSet patterns(context);
    patterns.add<UpdateAsyncWaitCount>(context);
    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUUpdateAsyncWaitCountPass(std::string archGenName) {
  return std::make_unique<TritonAMDGPUUpdateAsyncWaitCountPass>(archGenName);
}
