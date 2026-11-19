#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/MBarrierUtilities.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUINSERTREMOTESHAREDSTOREBARRIERSPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;

struct InsertRemoteSharedStoreBarriersPass
    : public impl::TritonNvidiaGPUInsertRemoteSharedStoreBarriersPassBase<
          InsertRemoteSharedStoreBarriersPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    if (computeCapability < 90 || ttg::TritonGPUDialect::getNumCTAs(mod) == 1)
      return;

    mod.walk([](Operation *op) {
      if (!isa<ttg::LocalStoreOp, ttg::LocalAllocOp>(op) ||
          !hasCrossCTASharedAccess(op).value_or(false))
        return;

      OpBuilder builder(op);
      Location loc = op->getLoc();
      // Rendezvous after each CTA's preceding accesses and async waits.
      ClusterBarrierOp::create(builder, loc);
      builder.setInsertionPointAfter(op);
      // Publish remote writes to generic and async consumers.
      FenceAsyncSharedOp::create(builder, loc, /*bCluster=*/true);
      ClusterBarrierOp::create(builder, loc);
    });
  }
};

} // namespace
} // namespace mlir::triton::nvidia_gpu
