#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
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

  auto sharedAttr = mod->getAttrOfType<IntegerAttr>("ttg.shared");
  int64_t shared = sharedAttr ? sharedAttr.getInt() : 0;
  int64_t nextOffset = llvm::alignTo(shared, int64_t{8});
  DenseMap<Region *, IntegerAttr> regionOffsets;
  Builder builder(mod.getContext());

  kernel.walk([&](Operation *op) {
    if (!needsClusterBarrier(op))
      return;
    if (!op->getParentOfType<gpu::WarpSpecializeOp>())
      return;
    Region *region = op->getParentRegion();
    while (!isa<gpu::WarpSpecializeOp, gpu::WarpSpecializePartitionsOp>(
        region->getParentOp()))
      region = region->getParentOp()->getParentRegion();

    auto [it, inserted] = regionOffsets.try_emplace(region);
    if (inserted) {
      it->second = builder.getI32IntegerAttr(nextOffset);
      nextOffset += kClusterBarrierMbarAllocationSize;
    }
    op->setAttr(kClusterBarrierMbarOffsetAttrName, it->second);
  });

  if (regionOffsets.empty())
    mod->removeAttr(kWSClusterBarrierCountAttrName);
  else {
    mod->setAttr(kWSClusterBarrierCountAttrName,
                 builder.getI32IntegerAttr(regionOffsets.size()));
    mod->setAttr("ttg.shared", builder.getI32IntegerAttr(nextOffset));
  }
}

} // namespace mlir::triton::nvidia_gpu
