#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/DescriptorUtils.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::nvidia_gpu {

static bool isTMACompatibleEncoding(Attribute enc) {
  if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(enc)) {
    return !nvmma.getTransposed();
  }
  return false;
}

Attribute findLoadEncodingFromUsers(Operation *op) {
  // Ignore multiple users and just pick the first compatible layout
  for (auto use : op->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(use)) {
      auto enc = alloc.getType().getEncoding();
      if (isTMACompatibleEncoding(enc))
        return enc;
    } else if (auto store = dyn_cast<ttg::LocalStoreOp>(use)) {
      auto enc = store.getDst().getType().getEncoding();
      if (isTMACompatibleEncoding(enc))
        return enc;
    }
  }
  return {};
}

Attribute getFallbackSharedEncoding(RankedTensorType tensorType,
                                    ttg::CGAEncodingAttr cgaLayout,
                                    ArrayRef<int64_t> usageShape,
                                    unsigned numCTAs) {
  auto ctx = tensorType.getContext();
  SmallVector<unsigned> order;
  for (int i = tensorType.getRank() - 1; i >= 0; --i)
    order.push_back(i);

  ArrayRef<int64_t> shape =
      usageShape.empty() ? tensorType.getShape() : usageShape;
  if (!cgaLayout) {
    // Arbitrarily distribute along the last dim
    SmallVector<unsigned> ctasPerCGA(tensorType.getRank(), 1);
    ctasPerCGA.back() = numCTAs;
    cgaLayout = ttg::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA,
                                                      ctasPerCGA, order);
  } else if (cgaLayout.getRank() != tensorType.getRank())
    cgaLayout = ttg::updateCGALayoutForShape(cgaLayout, shape);

  return ttg::NVMMASharedEncodingAttr::get(ctx, shape, order, cgaLayout,
                                           tensorType.getElementType(),
                                           /*fp4Padded*/ false);
}

bool isForcedToDefault(Operation *op) {
  return isa<CallOp, ReturnOp, ReinterpretTensorDescOp>(op);
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
    ttg::assignMemoryLayouts(m, findLoadEncodingFromUsers,
                             getFallbackSharedEncoding,
                             ttng::updateEncodingForShape, isForcedToDefault);
  }
};

} // namespace mlir::triton::nvidia_gpu
