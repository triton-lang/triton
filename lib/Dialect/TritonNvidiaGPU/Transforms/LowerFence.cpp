#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::nvidia_gpu {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONNVIDIAGPULOWERFENCEPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

class TritonNvidiaGPULowerFencePass
    : public impl::TritonNvidiaGPULowerFencePassBase<
          TritonNvidiaGPULowerFencePass> {
public:
  using impl::TritonNvidiaGPULowerFencePassBase<
      TritonNvidiaGPULowerFencePass>::TritonNvidiaGPULowerFencePassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool isMultiCTA = ttg::TritonGPUDialect::getNumCTAs(mod) > 1;
    SmallVector<FenceOp> fences;
    mod.walk([&](FenceOp fence) { fences.push_back(fence); });

    for (FenceOp fence : fences) {
      OpBuilder builder(fence);
      if (isMultiCTA)
        ttng::ClusterBarrierOp::create(builder, fence.getLoc());
      else
        ttg::BarrierOp::create(builder, fence.getLoc(), ttg::AddrSpace::All);
      fence.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
