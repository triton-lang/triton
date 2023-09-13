#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
//
// This pass works after all other passes, inserting fences to ensure that
// memory operations are properly ordered acorss genric and async proxy.
//
//===----------------------------------------------------------------------===//

using namespace mlir;
namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

using ::mlir::triton::gpu::SharedEncodingAttr;

namespace {

struct FenceInsertionPass
    : public TritonGPUFenceInsertionBase<FenceInsertionPass> {

public:
  FenceInsertionPass() = default;
  FenceInsertionPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }
  // TODO: support more patterns to insert fences
  // only support insertion between convert layout ops and dot ops to protect
  // flashattention
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    ModuleOp mod = getOperation();
    mod.walk([&](Operation *op) {
      if (isa<tt::DotOp>(op)) {
        auto a = op->getOperand(0);
        auto b = op->getOperand(1);
        auto mmaEncoding = op->getResult(0)
                               .getType()
                               .cast<RankedTensorType>()
                               .getEncoding()
                               .dyn_cast<ttg::MmaEncodingAttr>();
        auto isHopperEncoding = mmaEncoding && mmaEncoding.isHopper();
        if (isHopperEncoding && (isa<ttg::ConvertLayoutOp>(a.getDefiningOp()) &&
                                 ttg::isSharedEncoding(a)) ||
            (isa<ttg::ConvertLayoutOp>(b.getDefiningOp()) &&
             ttg::isSharedEncoding(b))) {

          // TODO: check whether cluster fence is needed
          OpBuilder builder(op);
          builder.create<ttng::FenceAsyncSharedOp>(op->getLoc(),
                                                   false /*bCluster*/);
        }
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUFenceInsertionPass(int computeCapability) {
  return std::make_unique<FenceInsertionPass>(computeCapability);
}
