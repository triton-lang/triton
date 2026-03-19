#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/DescriptorUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::nvidia_gpu {

static bool isTMACompatibleEncoding(Attribute enc) {
  if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(enc)) {
    return !nvmma.getTransposed();
  }
  return false;
}

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEDESCRIPTORENCODINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

class TritonNvidiaGPUOptimizeDescriptorEncodingPass
    : public impl::TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
          TritonNvidiaGPUOptimizeDescriptorEncodingPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
      TritonNvidiaGPUOptimizeDescriptorEncodingPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // Fallback shared encoding callback
    auto buildFallbackSharedEncoding =
        [](mlir::MLIRContext *ctx, ArrayRef<int64_t> shape,
           ArrayRef<unsigned> order, ttg::CGAEncodingAttr cgaLayout,
           Type elementType) {
          return ttg::NVMMASharedEncodingAttr::get(
              ctx, shape, order, cgaLayout, elementType, /*fp4Padded*/ false);
        };

    ttg::DescriptorAnalysisCallbacks callbacks;
    callbacks.isCompatibleSharedEncoding = isTMACompatibleEncoding;
    callbacks.buildFallbackSharedEncoding = buildFallbackSharedEncoding;
    ttg::AssignDescriptorMemoryLayouts assignMemoryLayouts(callbacks);
    assignMemoryLayouts.assignMemoryLayouts(m);
  }
};

} // namespace mlir::triton::nvidia_gpu
