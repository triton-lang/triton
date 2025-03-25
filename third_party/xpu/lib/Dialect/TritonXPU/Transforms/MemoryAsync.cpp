//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 by Kunlunxin. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-memory-async"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUMEMORYASYNC
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

class TritonXPUMemoryAsyncPass
    : public impl::TritonXPUMemoryAsyncBase<TritonXPUMemoryAsyncPass> {
public:
  using impl::TritonXPUMemoryAsyncBase<
      TritonXPUMemoryAsyncPass>::TritonXPUMemoryAsyncBase;

  TritonXPUMemoryAsyncPass() = default;
  TritonXPUMemoryAsyncPass(bool dumpFlag) { this->dumpFlag = dumpFlag; }

  void loadOpAsyncCheck(triton::xpu::LoadOp loadOp_1,
                        triton::xpu::LoadOp loadOp_2) {
    OpBuilder builder(loadOp_1);

    if (loadOp_1->getBlock() == loadOp_2->getBlock() &&
        loadOp_1->isBeforeInBlock(loadOp_2)) {
      auto gm2lmOp_1 = cast<triton::xpu::GM2LMOp>(loadOp_1->getPrevNode());
      gm2lmOp_1->setAttr("async", builder.getBoolAttr(true));
      loadOp_1->moveAfter(loadOp_2);
      if (dumpFlag)
        LLVM_DEBUG(llvm::dbgs() << "Memory Async Optimization Hit!\n");
    }
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    mod->walk([&](triton::xpu::GM2LMOp gm2lmOp_1) {
      // Pruning
      if (gm2lmOp_1.getAsync())
        return;

      auto loadOp_1 = cast<triton::xpu::LoadOp>(gm2lmOp_1->getNextNode());

      auto loadop_user_begin = loadOp_1->user_begin();
      auto loadop_user_end = loadOp_1->user_end();

      if (!loadOp_1->hasOneUse())
        return;

      llvm::TypeSwitch<Operation *>(*loadop_user_begin)
          .Case<XPU_VVECTORIZED_BINARY_OP, XPU_SVECTORIZED_BINARY_OP,
                triton::xpu::VvdivFOp>([&](auto vBinOp) {
            auto lhsOp =
                vBinOp.getLhs().template getDefiningOp<triton::xpu::LoadOp>();
            auto rhsOp =
                vBinOp.getRhs().template getDefiningOp<triton::xpu::LoadOp>();

            if (lhsOp && rhsOp) {
              triton::xpu::LoadOp loadOp_2 = lhsOp == loadOp_1 ? rhsOp : lhsOp;
              loadOpAsyncCheck(loadOp_1, loadOp_2);
            }
          })
          .Case<mlir::arith::AddFOp, mlir::arith::SubFOp, mlir::arith::MulFOp,
                mlir::arith::MaximumFOp, mlir::arith::DivFOp>([&](auto binOp) {
            auto lhsOp =
                binOp.getLhs().template getDefiningOp<triton::xpu::LoadOp>();
            auto rhsOp =
                binOp.getRhs().template getDefiningOp<triton::xpu::LoadOp>();

            if (lhsOp && rhsOp) {
              triton::xpu::LoadOp loadOp_2 = lhsOp == loadOp_1 ? rhsOp : lhsOp;
              loadOpAsyncCheck(loadOp_1, loadOp_2);
            }
          })
          .Case<mlir::triton::xpu::VSelectOp, mlir::arith::SelectOp>(
              [&](auto selectOp) {
                auto tv = selectOp.getTrueValue()
                              .template getDefiningOp<triton::xpu::LoadOp>();
                auto fv = selectOp.getFalseValue()
                              .template getDefiningOp<triton::xpu::LoadOp>();

                if (tv && fv) {
                  triton::xpu::LoadOp loadOp_2 = tv == loadOp_1 ? fv : tv;
                  loadOpAsyncCheck(loadOp_1, loadOp_2);
                }
              })
          .Default([&](auto &op) {
            if (dumpFlag) {
              LLVM_DEBUG({
                op->dump();
                llvm::dbgs() << "Unsupport Op For Memory Async Optimization\n";
              });
            }
          });
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
