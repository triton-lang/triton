#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace mlir::triton::nvidia_gpu {

#define GEN_PASS_DEF_TRITONGPUBUFFERREGIONPROXYFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

struct BufferRegionProxyFenceInsertionPass
    : public impl::TritonGPUBufferRegionProxyFenceInsertionBase<
          BufferRegionProxyFenceInsertionPass> {
  using impl::TritonGPUBufferRegionProxyFenceInsertionBase<
      BufferRegionProxyFenceInsertionPass>::
      TritonGPUBufferRegionProxyFenceInsertionBase;

  void runOnOperation() override {
    if (computeCapability < 90)
      return;
  }
};

} // namespace mlir::triton::nvidia_gpu
