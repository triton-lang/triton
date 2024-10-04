#include "mlir/Analysis/Liveness.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/MapVector.h"

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
using namespace triton::nvidia_gpu;

namespace {

static int32_t roundUp(int32_t val, int32_t step) {
  auto t = val + step - 1;
  return t - (t % step);
}

static void allocateGMem(Operation *parentOp) {
  MLIRContext *ctx = parentOp->getContext();
  OpBuilder builder(ctx);
  int32_t offset = 0;
  uint32_t largestAlignment = 1;

  // Dumb allocation that ignores liveness and makes no attempt to minimize
  // padding
  // TODO: Use a real algorithm
  parentOp->walk<WalkOrder::PostOrder>(
      [&](triton::nvidia_gpu::GlobalScratchAllocOp alloc) {
        auto nbytes = alloc.getNbytes();
        auto align = alloc.getAlignment();
        offset = roundUp(offset, align);
        alloc->setAttr("triton_nvidia_gpu.global_scratch_memory_offset",
                       builder.getI32IntegerAttr(offset));
        offset += nbytes;
        largestAlignment = std::max(largestAlignment, align);
      });
  int32_t totalMemorySize = roundUp(offset, largestAlignment);
  parentOp->setAttr("triton_nvidia_gpu.global_scratch_memory_size",
                    builder.getI32IntegerAttr(totalMemorySize));
  parentOp->setAttr("triton_nvidia_gpu.global_scratch_memory_alignment",
                    builder.getI32IntegerAttr(largestAlignment));
}

class TritonNvidiaGPUGlobalScratchAllocationPass
    : public TritonNvidiaGPUGlobalScratchAllocationPassBase<
          TritonNvidiaGPUGlobalScratchAllocationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // TODO: handle cases with multiple function with GlobalScratchAllocOp.
    allocateGMem(mod);
  }
};

} // namespace

namespace mlir {

std::unique_ptr<Pass> createTritonNvidiaGPUGlobalScratchAllocationPass() {
  return std::make_unique<TritonNvidiaGPUGlobalScratchAllocationPass>();
}

} // namespace mlir
