#include "TritonAMDGPUTransforms/Passes.h"
#include "amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"

// This pass updates the waitCount of `AsyncWait` Ops to represent the number of
// inflight async load operation between the async_wait and the definition of
// the AsyncToken, thus allowing to wait only on the dependent async loads
// allowing loads issued after to complete in the future.
// This also means we should never overestimate the value to ensure
// correctness; being conservative and underestimating is fine given that only
// affects performance
// For each async_wait we need to compute the minimum across all AsyncToken
// operands.
// For each token the minimum number of async transaction along it's
// def chain is deduced. A token can be copied when passing in as loop initial
// argument and yielded from a loop body in which case we need to take the
// minimum along both paths.
// We do not exit early if we encounter another async_wait along the def chain
// because the pipeliner will merge redundant waits for us already

namespace tt = triton;
namespace ttg = triton::gpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUUPDATEASYNCWAITCOUNT
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Returns the number of individual async load memory transactions when copy
// data from the given |srcTy| in global memory to the given |dstTy| in shared
// memory.
int getNumberOfLoadInstructions(RankedTensorType srcTy,
                                ttg::MemDescType dstTy) {
  LinearLayout srcLayout = tt::gpu::toLinearLayout(srcTy);
  LinearLayout sharedLayout;
  if (auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(
          dstTy.getEncoding())) {
    sharedLayout = paddedEnc.getLinearComponent();
  } else {
    sharedLayout = triton::gpu::toLinearLayout(dstTy);
  }
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

} // anonymous namespace

struct TritonAMDGPUUpdateAsyncWaitCountPass
    : impl::TritonAMDGPUUpdateAsyncWaitCountBase<
          TritonAMDGPUUpdateAsyncWaitCountPass> {
  using Base::Base;

  void runOnOperation() override {
    tt::AMD::TargetInfo targetInfo(archGenerationName);
    if (!isCDNA(targetInfo.getISAFamily())) {
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

} // namespace mlir
