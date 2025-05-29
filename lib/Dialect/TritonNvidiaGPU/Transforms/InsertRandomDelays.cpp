#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

//===----------------------------------------------------------------------===//
//
// This pass inserts random delays before or after asynchronous operations to
// expose race conditions.
//
//===----------------------------------------------------------------------===//

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUINSERTRANDOMDELAYS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

struct InsertRandomDelaysPass
    : public impl::TritonGPUInsertRandomDelaysBase<InsertRandomDelaysPass> {

public:
  using impl::TritonGPUInsertRandomDelaysBase<
      InsertRandomDelaysPass>::TritonGPUInsertRandomDelaysBase;
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());
    mod.walk([&](Operation *op) {
      // Possibly use traits here instead of if statements?
      if (isa<ttg::AsyncWaitOp, ClusterWaitOp, WarpGroupDotWaitOp,
              TMAStoreWaitOp>(op)) {
        // Insert random delay after asynchronous operation.
        rewriter.setInsertionPointAfter(op);
        rewriter.create<RandomDelayOp>(op->getLoc());
      } else if (isa<FenceAsyncSharedOp, ClusterArriveOp, InitBarrierOp,
                     InvalBarrierOp, BarrierExpectOp, ArriveBarrierOp,
                     AsyncTMACopyGlobalToLocalOp, AsyncTMACopyLocalToGlobalOp,
                     AsyncTMAReduceOp, AsyncTMAGatherOp, AsyncTMAScatterOp>(
                     op)) {
        // Insert random delay before asynchronous operation.
        rewriter.setInsertionPoint(op);
        rewriter.create<RandomDelayOp>(op->getLoc());
      }
    });
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
