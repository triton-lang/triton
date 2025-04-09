#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

// This pass updates the waitCount of `AsyncWait` Ops to represent the number of
// inflight async load operation between the async_wait and the definition of
// the async_tokens. This value can then be used to wait only on the dependent
// async loads before accessing the data allowing loads issues after to complete
// in the future. This also means we should never overestimate the value to
// ensure correctness wherease underestimating only affects performance.
// So for each async_wait we need to compute the minimum across all async_token
// operands.
// For each token the minimum number of async transaction along it's def chain
// is deduced. Note that a token can have multiple producers, e.g. if it's loop
// carried (prologue and loop body). Therefore all paths to all producers of the
// async_token have to be analyzed.
// Note that we do not exit early if we encounter another async_wait along the
// def chain because the pipeliner will merge redundant waits for us already

using namespace mlir;
namespace tt = triton;
namespace ttg = triton::gpu;

// Overload to get the number of loads from direct to lds memory layouts
int getNumberOfLoadInstructions(RankedTensorType srcTy,
                                ttg::MemDescType dstTy) {
  auto shape = srcTy.getShape();
  LinearLayout srcLayout = tt::gpu::toLinearLayout(shape, srcTy.getEncoding());
  LinearLayout sharedLayout =
      tt::gpu::toLinearLayout(shape, dstTy.getEncoding());
  LinearLayout srcToSharedLayout = srcLayout.invertAndCompose(sharedLayout);

  // On GFX9 we cannot split direct to lds loads into multiple ones because we
  // need coalesced writes. So we can divide the number of registers by the
  // contiguity to get the number of load instructions.
  int contig = srcToSharedLayout.getNumConsecutiveInOut();
  int numberOfRegisters = srcToSharedLayout.getInDimSize(
      StringAttr::get(srcTy.getContext(), "register"));
  int loadInstructionCount = std::max(1, numberOfRegisters / contig);
  return loadInstructionCount;
}

// The pipeliner always insert ops following an order of ttg.async_load ->
// [token] -> ttg.async_commit_group -> [token] -> ttg.async_wait. So here we
// scan the operands of ttg.async_commit_group to count the number of issued
// async load intrinsics.
int getNumberOfLoadInstructions(Operation *op) {
  if (isa<ttg::AsyncCommitGroupOp>(op)) {
    int count = 0;
    for (auto token : op->getOperands()) {
      auto defOp = token.getDefiningOp();
      if (!defOp)
        continue;
      if (auto copyOp = llvm::dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(defOp)) {
        count += getNumberOfLoadInstructions(copyOp.getSrc().getType(),
                                             copyOp.getResult().getType());
      } else if (auto copyOp =
                     llvm::dyn_cast<amdgpu::BufferLoadToLocalOp>(defOp)) {
        auto srcTy = cast<RankedTensorType>(LLVM::AMD::getPointerTypeWithShape(
            copyOp.getPtr(), copyOp.getOffsets()));
        count += getNumberOfLoadInstructions(srcTy, copyOp.getDest().getType());
      }
    }
    return count;
  }
  if (isa<tt::LoadOp, tt::StoreOp, amdgpu::BufferLoadToLocalOp,
          amdgpu::BufferStoreOp, tt::AtomicRMWOp, tt::AtomicCASOp,
          amdgpu::BufferAtomicRMWOp>(op)) {
    op->emitRemark("Global memory operation between async wait and "
                   "async_loads. This will hinder the interleaving of memory "
                   "operations and might impact performance.");
  }
  return 0;
}

// LLVM cannot infer the dependency between direct to lds (async) loads and
// the local reads between warps in a workgroup. As a workaround we update the
// waitcnt to represent the number of hardware instructions we are
// interleaving with. This allows us to manually emit the waitcnt during
// lowering.
void updateWaitCount(ttg::AsyncWaitOp waitOp, RewriterBase &rewriter) {
  int waitCnt = std::numeric_limits<int>::max();

  // AsyncWait can await multiple tokens so we get the minimum from all
  // tokens
  for (auto token : waitOp.getOperands()) {
    // Traverse def chain from waitOp to the producer of the token and count
    // the minumum number of vmcnt instructions
    auto tokenWaitCnt =
        deduceMinCountOnDefChain(token, waitOp, [](Operation *op) {
          return getNumberOfLoadInstructions(op);
        });
    waitCnt = std::min(waitCnt, tokenWaitCnt);
  }

  if (waitCnt == std::numeric_limits<int>::max() || waitOp.getNum() == waitCnt)
    return;

  rewriter.modifyOpInPlace(waitOp, [&]() { waitOp.setNum(waitCnt); });
}

struct TritonAMDGPUUpdateAsyncWaitCountPass
    : public TritonAMDGPUUpdateAsyncWaitCountBase<
          TritonAMDGPUUpdateAsyncWaitCountPass> {
  TritonAMDGPUUpdateAsyncWaitCountPass(StringRef archGenName) {
    this->archGenerationName = archGenName.str();
  }

  void runOnOperation() override {
    tt::AMD::TargetInfo targetInfo(archGenerationName);
    if (!targetInfo.isCDNA()) {
      return;
    }

    ModuleOp m = getOperation();

    SmallVector<ttg::AsyncWaitOp> waitOps;
    getOperation()->walk(
        [&](ttg::AsyncWaitOp waitOp) { waitOps.push_back(waitOp); });

    for (auto waitOp : waitOps) {
      IRRewriter builder(waitOp->getContext());
      updateWaitCount(waitOp, builder);
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonAMDGPUUpdateAsyncWaitCountPass(std::string archGenName) {
  return std::make_unique<TritonAMDGPUUpdateAsyncWaitCountPass>(archGenName);
}
