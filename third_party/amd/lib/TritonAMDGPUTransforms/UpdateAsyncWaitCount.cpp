
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
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

// LLVM cannot infer the dependency between direct to lds (async) loads and the
// local reads between warps in a workgroup. As a workaround we update the
// waitcnt to represent the number of hardware instructions we are interleaving
// with. This allows us to manually emit the waitcnt when lowering.
struct UpdateAsyncWaitCount : public OpRewritePattern<ttg::AsyncWaitOp> {
  UpdateAsyncWaitCount(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(ttg::AsyncWaitOp waitOp,
                                PatternRewriter &rewriter) const override {
    int waitCnt = std::numeric_limits<int>::max();

    // AsyncWait can wait on multiple tokens we have to use the minimum as our
    // waitcnt
    for (auto token : waitOp.getOperands()) {
      // Traverse def chain from waitOp to the producer of the token and count
      // the minumum number of vmcnt instructions
      auto tokenWaitCnt =
          findMinCountInDefChain(token, waitOp, [this](Operation *op) {
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

  // Computes (conservatively) the number of vmcnt instructions the op will
  // produce
  int getNumberOfLoadInstructions(Operation *op) const {
    if (isa<ttg::AsyncCommitGroupOp>(op)) {
      Value token = op->getOperand(0);
      auto defToken = token.getDefiningOp();
      if (defToken) {
        return llvm::TypeSwitch<Operation *, unsigned>(defToken)
            .Case([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
              return getNumberOfLoadInstructions(getContext(),
                                                 copyOp.getSrc().getType(),
                                                 copyOp.getResult().getType());
            })
            .Case([&](amdgpu::BufferLoadToLocalOp copyOp) {
              RankedTensorType srcTy =
                  cast<RankedTensorType>(LLVM::AMD::getPointerTypeWithShape(
                      copyOp.getPtr(), copyOp.getOffsets()));
              return getNumberOfLoadInstructions(getContext(), srcTy,
                                                 copyOp.getDest().getType());
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
    return 0;
  }

  // Overload to get the number of loads from direct to lds loads
  int getNumberOfLoadInstructions(MLIRContext *context, RankedTensorType srcTy,
                                  ttg::MemDescType dstTy) const {
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

    patterns.add<UpdateAsyncWaitCount>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUUpdateAsyncWaitCountPass(std::string archGenName) {
  return std::make_unique<TritonAMDGPUUpdateAsyncWaitCountPass>(archGenName);
}
