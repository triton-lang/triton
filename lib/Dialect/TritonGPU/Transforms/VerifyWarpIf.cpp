#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUVERIFYWARPIF
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct VerifyWarpIf : impl::TritonGPUVerifyWarpIfBase<VerifyWarpIf> {
  using TritonGPUVerifyWarpIfBase::TritonGPUVerifyWarpIfBase;
  void runOnOperation() override {
    WalkResult result = getOperation().walk([&](WarpIfOp op) {
      return failed(op.verifyBody(allowFlushDenorm)) ? WalkResult::interrupt()
                                                     : WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace mlir::triton::gpu
