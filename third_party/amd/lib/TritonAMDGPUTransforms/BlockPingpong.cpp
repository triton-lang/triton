#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

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
  SmallVector<SmallVector<Value>> loadSlices;
  SmallVector<Operation *> dotSlices;

  IntegerAttr schedMaskAttr;
  IntegerAttr schedMaskAttr0;
  IntegerAttr lowPrioAttr;
  IntegerAttr highPrioAttr;

public:
  Pingponger(scf::ForOp forOp) : forOp(forOp){};
  void getDotPingponged();
  LogicalResult genLocalSlice(OpBuilder &builder, Value v,
                              Attribute dotEncoding, unsigned opIdx,
                              unsigned numSlices);
  void transformOneWave(OpBuilder &builder, Location loc);
  void transformTwoWave(OpBuilder &builder, Location loc);
};

// Transform each wave coming from the different blocks
void Pingponger::transformOneWave(OpBuilder &builder, Location loc) {
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
  auto setPrioBack0 = builder.create<ROCDL::SetPrioOp>(loc, lowPrioAttr);
  setPrio1->moveBefore(dotOps[0]);
  setPrioBack0->moveAfter(dotOps[0]);
}

LogicalResult Pingponger::genLocalSlice(OpBuilder &builder, Value v,
                                        Attribute dotEncoding, unsigned opIdx,
                                        unsigned numSlices) {
  SmallVector<Value> slices;
  auto type = cast<triton::MemDescType>(v.getType());
  auto srcEncoding =
      mlir::cast<ttg::DotOperandEncodingAttr>(type.getEncoding());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  Type elementType = type.getElementType();

  int64_t kIdx = opIdx == 0 ? 1 : 0;
  int64_t sliceWidth = shape[kIdx] / numSlices;
  if (sliceWidth < 16)
    return failure();

  for (int i = 0; i < numSlices; i++) {
    int64_t offset = sliceWidth * i;
    SmallVector<Value> offsetsVal;
    offsetsVal.push_back(
        builder.create<arith::ConstantIntOp>(v.getLoc(), 0, 32));
    offsetsVal.push_back(
        builder.create<arith::ConstantIntOp>(v.getLoc(), offset, 32));
    Value newSmem = builder.create<triton::gpu::MemDescSubviewOp>(
        v.getLoc(),
        triton::MemDescType::get(shape, elementType, type.getEncoding(),
                                 type.getMemorySpace()),
        v, offsetsVal);
    auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
        builder.getContext(), opIdx, dotEncoding, srcEncoding.getKWidth());
    Value prefetchSlice = builder.create<triton::gpu::LocalLoadOp>(
        v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
        newSmem);
    slices.push_back(prefetchSlice);
  }

  loadSlices.push_back(slices);
  return success();
}

// Transform two waves coming from the same blocks
void Pingponger::transformTwoWave(OpBuilder &builder, Location loc) {
  // Schedule operations as below
  // mem1: global load A, local load A(1/4), local load B(1/4)
  // dot (1/4)
  // mem2: global load B, local load A(2/4), local load B(2/4)
  // dot (2/4)
  // mem3: local load A(3/4, 4/4), local load B(3/4, 4/4)
  // dot (3/4)
  // mem3: local store A and B
  // dot (4/4)

  // First, slice local_loads and dot into 4 parts
  auto op = cast<triton::DotOp>(dotOps[0]);
  auto dotEncoding = op.getType().getEncoding();
  if (genLocalSlice(builder, op.getA(), dotEncoding, 0, 4).failed() ||
      genLocalSlice(builder, op.getB(), dotEncoding, 1, 4).failed())
    return;

  // Clone dots four times and use slices
  

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
  auto setPrioBack0 = builder.create<ROCDL::SetPrioOp>(loc, lowPrioAttr);
  setPrio1->moveBefore(dotOps[0]);
  setPrioBack0->moveAfter(dotOps[0]);
}

void Pingponger::getDotPingponged() {

  OpBuilder builder(forOp);
  MLIRContext *ctx = forOp.getContext();
  Location loc = forOp.getLoc();

  schedMaskAttr = IntegerAttr::get(IntegerType::get(ctx, 32), 6);
  schedMaskAttr0 = IntegerAttr::get(IntegerType::get(ctx, 32), 0);
  lowPrioAttr = IntegerAttr::get(IntegerType::get(ctx, 16), 0);
  highPrioAttr = IntegerAttr::get(IntegerType::get(ctx, 16), 1);
  auto f16_ty = builder.getF16Type();

  forOp->walk([&](Operation *op) {
    if (isa<triton::LoadOp>(op))
      gLoadOps.push_back(op);
    if (isa<ttg::LocalLoadOp>(op)) {
      // Check if this is loading from pipeline
      auto src = op->getOperand(0);
      if (auto arg = mlir::dyn_cast<BlockArgument>(src))
        if (forOp.getTiedLoopInit(arg)->get())
          lLoadOps.push_back(op);
    }
    if (isa<triton::DotOp>(op))
      dotOps.push_back(op);
  });

  if (gLoadOps.size() != 2 || lLoadOps.size() != 2 || dotOps.size() != 1)
    return;

  transformOneWave(builder, loc);
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
