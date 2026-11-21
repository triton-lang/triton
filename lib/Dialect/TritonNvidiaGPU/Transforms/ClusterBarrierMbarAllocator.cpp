#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {
#define GEN_PASS_DEF_TRITONNVIDIAGPUCLUSTERBARRIERMBARALLOCATORPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace mlir::triton::nvidia_gpu {

namespace {

struct ClusterBarrierMbarAllocatorPass
    : public impl::TritonNvidiaGPUClusterBarrierMbarAllocatorPassBase<
          ClusterBarrierMbarAllocatorPass> {
  void runOnOperation() override {
    runClusterBarrierMbarAllocator(getOperation());
  }
};

} // namespace

void runClusterBarrierMbarAllocator(ModuleOp mod) {
  auto funcs = mod.getOps<triton::FuncOp>();
  auto kernelIt = llvm::find_if(
      funcs, [](triton::FuncOp func) { return triton::isKernel(func); });
  if (kernelIt == funcs.end())
    return;
  triton::FuncOp kernel = *kernelIt;

  auto launches = kernel.getOps<gpu::WarpSpecializeOp>();
  if (launches.empty()) {
    mod->removeAttr(kWSClusterBarrierCountAttrName);
    return;
  }

  auto sharedAttr = mod->getAttrOfType<IntegerAttr>("ttg.shared");
  int64_t shared = sharedAttr ? sharedAttr.getInt() : 0;
  int64_t base = llvm::alignTo(shared, int64_t{8});
  Builder builder(mod.getContext());
  auto assignSlot = [&](Region &region, unsigned slot) {
    bool used = false;
    region.walk([&](Operation *op) {
      if (!needsClusterBarrier(op))
        return;
      int64_t offset = base + slot * kClusterBarrierMbarAllocationSize;
      op->setAttr(kClusterBarrierMbarOffsetAttrName,
                  builder.getI32IntegerAttr(offset));
      used = true;
    });
    return used;
  };

  // Callees add the calling role's slot index to the base during lowering.
  bool calleeNeedsSlot = false;
  for (triton::FuncOp fn : funcs)
    if (fn != kernel)
      calleeNeedsSlot |= assignSlot(fn.getBody(), 0);

  // The default role also calls functions outside of warp specialization.
  unsigned count = calleeNeedsSlot ? 1 : 0;
  for (auto ws : launches) {
    SmallVector<Region *> roles{&ws.getDefaultRegion()};
    llvm::append_range(roles, ws.getPartitionRegions());
    for (auto [index, role] : llvm::enumerate(roles)) {
      unsigned slot = calleeNeedsSlot ? index : count;
      if (assignSlot(*role, slot) ||
          (calleeNeedsSlot && !role->getOps<CallOpInterface>().empty()))
        count = std::max(count, slot + 1);
    }
  }

  if (!count)
    mod->removeAttr(kWSClusterBarrierCountAttrName);
  else {
    mod->setAttr(kWSClusterBarrierCountAttrName,
                 builder.getI32IntegerAttr(count));
    mod->setAttr("ttg.shared",
                 builder.getI32IntegerAttr(
                     base + count * kClusterBarrierMbarAllocationSize));
  }
}

} // namespace mlir::triton::nvidia_gpu
