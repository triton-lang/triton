#include "Conversion/ProtonGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/Utility.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton::proton::gpu {
#define GEN_PASS_DEF_ADDSCHEDBARRIERS
#include "Conversion/ProtonGPUToLLVM/Passes.h.inc"
} // namespace triton::proton::gpu
} // namespace mlir

namespace {

struct AddSchedBarriers
    : public mlir::triton::proton::gpu::impl::AddSchedBarriersBase<
          AddSchedBarriers> {

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);

    auto funcOps = triton::proton::gpu::getTritonFunctions(mod);
    assert(funcOps.size() == 1 && "Expected exactly one funcOp");

    IntegerAttr zeroAttrValue =
        builder.getI32IntegerAttr(static_cast<int32_t>(0));

    funcOps[0].walk([&](mlir::triton::proton::gpu::ReadCounterOp op) {
      auto loc = op.getLoc();
      if (!isa_and_nonnull<ROCDL::SchedBarrier>(op->getPrevNode())) {
        builder.setInsertionPoint(op);
        builder.create<ROCDL::SchedBarrier>(loc, zeroAttrValue);
      }
    });

    funcOps[0].walk([&](mlir::triton::proton::gpu::CircularStoreOp op) {
      auto loc = op.getLoc();
      if (!isa_and_nonnull<ROCDL::SchedBarrier>(op->getNextNode())) {
        builder.setInsertionPointAfter(op);
        builder.create<ROCDL::SchedBarrier>(loc, zeroAttrValue);
      }
    });
  }
};

} // namespace

namespace mlir::triton::proton::gpu {

std::unique_ptr<OperationPass<ModuleOp>> createAddSchedBarriersPass() {
  return std::make_unique<AddSchedBarriers>();
}

} // namespace mlir::triton::proton::gpu
