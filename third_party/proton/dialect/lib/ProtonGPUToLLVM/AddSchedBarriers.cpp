#include "Conversion/ProtonGPUToLLVM/Passes.h"
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

    int numFuncOps = 0;
    FunctionOpInterface func;
    mod.walk([&](FunctionOpInterface op) {
      // Ignore any intrinsic functions. On AMD the predicate load/store ops
      // are currently pseduo instrunctions at this point and may get picked up
      // here and trigger the FunctionOpInterface range based assert below
      StringRef funcName(op.getNameAttr());
      if (!funcName.contains("__")) {
        numFuncOps += 1;
        func = op;
      }
    });

    assert(numFuncOps == 1);

    IntegerAttr zeroAttrValue =
        builder.getI32IntegerAttr(static_cast<int32_t>(0));

    func.walk([&](mlir::triton::proton::gpu::ReadCounterOp op) {
      auto loc = op.getLoc();
      if (!isa_and_nonnull<ROCDL::SchedBarrier>(op->getPrevNode())) {
        builder.setInsertionPoint(op);
        builder.create<ROCDL::SchedBarrier>(loc, zeroAttrValue);
      }
    });

    func.walk([&](mlir::triton::proton::gpu::CircularStoreOp op) {
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
