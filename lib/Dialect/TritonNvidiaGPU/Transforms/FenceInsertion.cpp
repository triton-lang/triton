#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/Sys/GetEnv.hpp"
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
  // TODO: support more general patterns to insert fences. eg. any op(generic)
  // to shared in use-def chain which refers by async proxy. We have generic(
  // convertlayout with sts/stmatix) + fence + async(wgmma/tma store) up to now
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    // ENABLE_MMA_V3
    if (!::triton::tools::getBoolEnv("ENABLE_MMA_V3"))
      return;
    ModuleOp mod = getOperation();
    mod.walk([&](Operation *op) {
      if (isa<tt::DotOp, ttng::DotAsyncOp>(op)) {
        OpBuilder builder(op);
        auto a = op->getOperand(0);
        auto b = op->getOperand(1);
        auto mmaEncoding = op->getResult(0)
                               .getType()
                               .cast<RankedTensorType>()
                               .getEncoding()
                               .dyn_cast<ttg::MmaEncodingAttr>();
        auto isHopperEncoding = mmaEncoding && mmaEncoding.isHopper();
        if (isHopperEncoding && (canReachGeneric(a) || canReachGeneric(b))) {
          builder.create<ttng::FenceAsyncSharedOp>(op->getLoc(),
                                                   false /*bCluster*/);
        }
      }
    });
  }

private:
  bool canReachGeneric(Value operand) {
    auto op = operand.getDefiningOp();

    if (BlockArgument arg = dyn_cast<BlockArgument>(operand)) {
      unsigned argNum = arg.getArgNumber();
      Operation *argOwner = arg.getOwner()->getParentOp();
      // suport ForOp
      if (auto forOp = dyn_cast<scf::ForOp>(argOwner)) {
        Value v = forOp.getBody()->getTerminator()->getOperand(argNum - 1);
        if (canReachGeneric(v))
          return true;
      }
    }

    if (!op)
      return false;
    if (isa<ttg::ConvertLayoutOp>(op) && ttg::isSharedEncoding(operand))
      return true;
    for (auto v : op->getOperands()) {
      if (canReachGeneric(v))
        return true;
    }
    return false;
  }
};

} // namespace

std::unique_ptr<Pass>
mlir::createTritonNvidiaGPUFenceInsertionPass(int computeCapability) {
  return std::make_unique<FenceInsertionPass>(computeCapability);
}
