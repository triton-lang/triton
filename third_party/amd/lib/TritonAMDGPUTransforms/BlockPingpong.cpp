#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
//#include "mlir/IR/Dominance.h"
//#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
//#include <deque>

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

class Pingponger {
  scf::ForOp forOp;
  SmallVector<Operation *> gLoadOps;
  SmallVector<Operation *> lLoadOps;
  SmallVector<Operation *> dotOps;

public:
  Pingponger() = delete;
  Pingponger(scf::ForOp forOp) : forOp(forOp) {};
  void getDotPingponged();
};

void Pingponger::getDotPingponged() {

  OpBuilder builder(forOp);
  MLIRContext *ctx = forOp.getContext();
  Location loc = forOp.getLoc();
  IntegerAttr schedMaskAttr = IntegerAttr::get(IntegerType::get(ctx, 32), 6);
  IntegerAttr schedMaskAttr0 = IntegerAttr::get(IntegerType::get(ctx, 32), 0);
  IntegerAttr lowPrioAttr = IntegerAttr::get(IntegerType::get(ctx, 16), 0);
  IntegerAttr highPrioAttr = IntegerAttr::get(IntegerType::get(ctx, 16), 1);
  auto f16_ty = builder.getF16Type();

  forOp.walk([&](triton::LoadOp op) {
    auto ty = cast<RankedTensorType>(op->getResultTypes()[0]);
    auto shape = ty.getShape();
    if (shape.size() != 2)
      return;
    // Only support 128x128x64xf16 at the moment.
    if (ty.getElementType() == f16_ty && shape[0] * shape[1] == 8192 &&
        shape[0] + shape[1] == 192)
      gLoadOps.push_back(op);
  });

  forOp.walk([&](ttg::LocalLoadOp op) {
    auto ty = cast<RankedTensorType>(op->getResultTypes()[0]);
    auto shape = ty.getShape();
    if (shape.size() != 2)
      return;
    // Only support 128x128x64xf16 at the moment.
    if (ty.getElementType() == f16_ty && shape[0] * shape[1] == 8192 &&
        shape[0] + shape[1] == 192)
      lLoadOps.push_back(op);
  });

  forOp.walk([&](triton::DotOp op) {
    auto ty = cast<RankedTensorType>(op->getResultTypes()[0]);
    auto shape = ty.getShape();
    if (shape.size() != 2)
      return;
    // Only support 128x128x64xf16 at the moment.
    if (ty.getElementType() == f16_ty && shape[0] == 128 && shape[1] == 128)
      dotOps.push_back(op);
  });

  if (gLoadOps.size() != 2 || lLoadOps.size() != 2 || dotOps.size() != 1)
    return;

  // Set low priority for the memory ops
  auto setPrio0 = builder.create<ROCDL::SetPrioOp>(loc, lowPrioAttr);
  setPrio0->moveBefore(gLoadOps[0]);

  // Splitting loading A and B inorder to prevent global/local load units
  // from the congestion.
  // Locate global load at the end. Otherwise, local_load at the end of
  // the sequence will be overlapped with the first local_stores from
  // the other warp.
  // sched.barriers to keep the order.
  lLoadOps[0]->moveBefore(gLoadOps[0]);
  auto schedB0 = builder.create<ROCDL::SchedBarrier>(loc, schedMaskAttr);
  schedB0->moveAfter(lLoadOps[0]);
  auto schedB1 = builder.create<ROCDL::SchedBarrier>(loc, schedMaskAttr);
  schedB1->moveAfter(gLoadOps[0]);
  lLoadOps[1]->moveBefore(gLoadOps[1]);
  auto schedB2 = builder.create<ROCDL::SchedBarrier>(loc, schedMaskAttr);
  schedB2->moveAfter(lLoadOps[1]);
  auto schedB3 = builder.create<ROCDL::SchedBarrier>(loc, schedMaskAttr0);
  schedB3->moveAfter(gLoadOps[1]);
  auto schedB4 = builder.create<ROCDL::SchedBarrier>(loc, schedMaskAttr0);
  schedB4->moveAfter(dotOps[0]);
  auto setPrio1 = builder.create<ROCDL::SetPrioOp>(loc, highPrioAttr);
  setPrio1->moveBefore(dotOps[0]);
}

class TritonAMDGPUBlockPingpongPass
    : public TritonAMDGPUBlockPingpongBase<TritonAMDGPUBlockPingpongPass> {
public:
  TritonAMDGPUBlockPingpongPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    int numWarps = ttg::TritonGPUDialect::getNumWarps(m);
    if (numWarps != 4)
      return;

    m.walk([&](scf::ForOp forOp) {
      Pingponger pingponger(forOp);
      pingponger.getDotPingponged();
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUBlockPingpongPass() {
  return std::make_unique<TritonAMDGPUBlockPingpongPass>();
}
