#include "TritonAMDGPUTransforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

#define GEN_PASS_DEF_TRITONAMDGPULOWERFENCE
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

class TritonAMDGPULowerFencePass
    : public impl::TritonAMDGPULowerFenceBase<TritonAMDGPULowerFencePass> {
public:
  using impl::TritonAMDGPULowerFenceBase<
      TritonAMDGPULowerFencePass>::TritonAMDGPULowerFenceBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool isMultiCTA = ttg::TritonGPUDialect::getNumCTAs(mod) > 1;
    SmallVector<tt::FenceOp> fences;
    mod.walk([&](tt::FenceOp fence) { fences.push_back(fence); });

    // TODO: Lower to a strong AMD cluster barrier once cluster-scoped memory
    // ordering is implemented.
    if (isMultiCTA && !fences.empty()) {
      fences.front().emitError(
          "AMD multi-CTA fence lowering is not implemented");
      return signalPassFailure();
    }

    for (tt::FenceOp fence : fences) {
      OpBuilder builder(fence);
      ttg::BarrierOp::create(builder, fence.getLoc(), ttg::AddrSpace::All);
      fence.erase();
    }
  }
};

} // namespace
} // namespace mlir
