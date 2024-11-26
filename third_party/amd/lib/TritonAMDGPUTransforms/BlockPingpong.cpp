#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

class Pingponger {
  scf::ForOp forOp;
  SmallVector<Operation *> gLoadOps;
  SmallVector<Operation *> lLoadOps;
  SmallVector<Operation *> lStoreOps;
  SmallVector<Operation *> dotOps;
  SmallVector<SmallVector<Operation *>> subViewOps;
  SmallVector<SmallVector<Operation *>> loadSliceOps;
  SmallVector<Operation *> dotSliceOps;
  SmallVector<Value> constOffsets;

  Operation *lastInsertedOp;

  IntegerAttr schedMaskAttr;
  IntegerAttr schedMaskAttr0;
  IntegerAttr lowPrioAttr;
  IntegerAttr highPrioAttr;

  int32_t kWidth;
  int32_t numWarps;

public:
  Pingponger(scf::ForOp forOp, int32_t numWarps)
      : forOp(forOp), numWarps(numWarps) {};
  void getDotPingponged();
  void genOffsetConstants(Location loc, OpBuilder &builder, unsigned numSlices,
                          int64_t sliceWidth);
  LogicalResult genLocalSlice(OpBuilder &builder, Value v,
                              Attribute dotEncoding, unsigned opIdx,
                              unsigned numSlices, int64_t sliceWidth);
  void transformOneWarp(OpBuilder &builder, Location loc);
  void transformTwoWarp(OpBuilder &builder, Location loc);
  void initOpInsertion(Operation *Op);
  void attachOp(Operation *Op);
};

void Pingponger::initOpInsertion(Operation *op) { lastInsertedOp = op; }

void Pingponger::attachOp(Operation *op) {
  assert(lastInsertedOp != nullptr);
  op->moveAfter(lastInsertedOp);
  lastInsertedOp = op;
}

// Transform each warp coming from the different blocks
void Pingponger::transformOneWarp(OpBuilder &builder, Location loc) {
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

void Pingponger::genOffsetConstants(Location loc, OpBuilder &builder,
                                    unsigned numSlices, int64_t sliceWidth) {
  for (int i = 0; i < numSlices; i++) {
    int64_t offset = sliceWidth * i;
    constOffsets.push_back(
        builder.create<arith::ConstantIntOp>(loc, offset, 32));
  }
}

LogicalResult Pingponger::genLocalSlice(OpBuilder &builder, Value v,
                                        Attribute dotEncoding, unsigned opIdx,
                                        unsigned numSlices,
                                        int64_t sliceWidth) {
  SmallVector<Operation *> slices;
  SmallVector<Operation *> subviews;
  auto memDesc = v.getDefiningOp()->getOperand(0);
  auto type = cast<ttg::MemDescType>(memDesc.getType());
  auto encoding = cast<RankedTensorType>(v.getType()).getEncoding();
  auto srcEncoding = mlir::cast<ttg::DotOperandEncodingAttr>(encoding);
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  Type elementType = type.getElementType();

  int64_t kIdx = opIdx == 0 ? 1 : 0;
  shape[kIdx] = sliceWidth;
  if (sliceWidth < 16)
    return failure();

  for (int i = 0; i < numSlices; i++) {

    SmallVector<Value> offsetsVal;
    SmallVector<int64_t> offsets = {0, 0};
    offsets[kIdx] = i;
    for (int64_t off : offsets) {
      offsetsVal.push_back(constOffsets[off]);
    }

    Value newSmem = builder.create<triton::gpu::MemDescSubviewOp>(
        v.getLoc(),
        ttg::MemDescType::get(shape, elementType, type.getEncoding(),
                                 type.getMemorySpace()),
        memDesc, offsetsVal);
    auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
        builder.getContext(), opIdx, dotEncoding, srcEncoding.getKWidth());
    Value prefetchSlice = builder.create<triton::gpu::LocalLoadOp>(
        v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
        newSmem);
    subviews.push_back(newSmem.getDefiningOp());
    slices.push_back(prefetchSlice.getDefiningOp());
  }

  subViewOps.push_back(subviews);
  loadSliceOps.push_back(slices);
  return success();
}

// Transform two warps coming from the same blocks
void Pingponger::transformTwoWarp(OpBuilder &builder, Location loc) {

  // First, slice local_loads and dot into 4 parts
  unsigned numSlices = 4;
  auto op = cast<triton::DotOp>(dotOps[0]);

  builder.setInsertionPointToStart(forOp.getBody());
  auto typeB = op.getB().getType();
  auto shapeB = typeB.getShape();
  int64_t sliceWidth = shapeB[0] / numSlices;
  genOffsetConstants(loc, builder, numSlices, sliceWidth);

  builder.setInsertionPointAfter(gLoadOps[0]);
  auto dotEncoding = op.getType().getEncoding();
  if (genLocalSlice(builder, op.getA(), dotEncoding, 0, numSlices, sliceWidth)
          .failed() ||
      genLocalSlice(builder, op.getB(), dotEncoding, 1, numSlices, sliceWidth)
          .failed())
    return;

  // Clone dots four times to consume the slices
  Operation *prevDot = op;
  for (int i = 0; i < numSlices; i++) {
    auto newOp = builder.clone(*op);
    newOp->setOperand(0, loadSliceOps[0][i]->getResult(0));
    newOp->setOperand(1, loadSliceOps[1][i]->getResult(0));
    if (i > 0)
      newOp->setOperand(2, prevDot->getResult(0));
    prevDot = newOp;
    dotSliceOps.push_back(newOp);
  }
  op->replaceAllUsesWith(prevDot);
  op->erase();
  lLoadOps[0]->erase();
  lLoadOps[1]->erase();

  SmallVector<Operation *> schedBarriers;
  SmallVector<Operation *> barriers;
  SmallVector<Operation *> setPrioHighs;
  SmallVector<Operation *> setPrioLows;

  for (int i = 0; i < 8; i++) {
    schedBarriers.push_back(
        builder.create<ROCDL::SchedBarrier>(loc, schedMaskAttr0));
    // MembarAnalysis can recognize gpu::BarrierOp and skip inserting additional
    // barrier
    barriers.push_back(builder.create<gpu::BarrierOp>(loc));

    if (i < 4) { // only required by dots
      setPrioHighs.push_back(
          builder.create<ROCDL::SetPrioOp>(loc, highPrioAttr));
      setPrioLows.push_back(builder.create<ROCDL::SetPrioOp>(loc, lowPrioAttr));
    }
  }

  // Reorder operations as below scheduling
  // mem1: global load A, local load A(1/4), local load B(1/4)
  // dot (1/4)
  // mem2: global load B, local load A(2/4), local load B(2/4)
  // dot (2/4)
  // mem3: local load A(3/4, 4/4), local load B(3/4, 4/4)
  // dot (3/4)
  // mem3: local store A and B
  // dot (4/4)

  // mem1: global load A, local load A(1/4), local load B(1/4)
  initOpInsertion(gLoadOps[1]);
  attachOp(subViewOps[0][0]);
  attachOp(subViewOps[1][0]);
  attachOp(loadSliceOps[0][0]);
  attachOp(loadSliceOps[1][0]);
  attachOp(barriers.back());
  attachOp(schedBarriers.back());

  schedBarriers.pop_back();
  barriers.pop_back();

  // dot (1/4)
  attachOp(setPrioHighs.back());
  attachOp(dotSliceOps[0]);
  attachOp(setPrioLows.back());
  attachOp(barriers.back());
  attachOp(schedBarriers.back());

  setPrioHighs.pop_back();
  setPrioLows.pop_back();
  schedBarriers.pop_back();
  barriers.pop_back();

  // mem2: global load B, local load A(2/4), local load B(2/4)
  attachOp(gLoadOps[1]);
  attachOp(subViewOps[0][1]);
  attachOp(subViewOps[1][1]);
  attachOp(loadSliceOps[0][1]);
  attachOp(loadSliceOps[1][1]);
  attachOp(barriers.back());
  attachOp(schedBarriers.back());

  schedBarriers.pop_back();
  barriers.pop_back();

  // dot (2/4)
  attachOp(setPrioHighs.back());
  attachOp(dotSliceOps[1]);
  attachOp(setPrioLows.back());
  attachOp(barriers.back());
  attachOp(schedBarriers.back());

  setPrioHighs.pop_back();
  setPrioLows.pop_back();
  schedBarriers.pop_back();
  barriers.pop_back();

  // mem3: local load A(3/4, 4/4), local load B(3/4, 4/4)
  attachOp(subViewOps[0][2]);
  attachOp(subViewOps[1][2]);
  attachOp(loadSliceOps[0][2]);
  attachOp(loadSliceOps[1][2]);
  attachOp(subViewOps[0][3]);
  attachOp(subViewOps[1][3]);
  attachOp(loadSliceOps[0][3]);
  attachOp(loadSliceOps[1][3]);
  attachOp(barriers.back());
  attachOp(schedBarriers.back());

  schedBarriers.pop_back();
  barriers.pop_back();

  // dot (3/4)
  attachOp(setPrioHighs.back());
  attachOp(dotSliceOps[2]);
  attachOp(setPrioLows.back());
  attachOp(barriers.back());
  attachOp(schedBarriers.back());

  setPrioHighs.pop_back();
  setPrioLows.pop_back();
  schedBarriers.pop_back();
  barriers.pop_back();

  // mem3: local store A and B
  initOpInsertion(lStoreOps[1]);
  attachOp(barriers.back());
  attachOp(schedBarriers.back());

  schedBarriers.pop_back();
  barriers.pop_back();

  // dot (4/4)
  attachOp(setPrioHighs.back());
  attachOp(dotSliceOps[3]);
  attachOp(setPrioLows.back());
  attachOp(barriers.back());
  attachOp(schedBarriers.back());

  setPrioHighs.pop_back();
  setPrioLows.pop_back();
  schedBarriers.pop_back();
  barriers.pop_back();
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
    if (isa<ttg::LocalStoreOp>(op))
      lStoreOps.push_back(op);
    if (isa<triton::DotOp>(op))
      dotOps.push_back(op);
  });

  if (gLoadOps.size() != 2 || lLoadOps.size() != 2 || dotOps.size() != 1)
    return;

  if (numWarps == 4) { // pingpong between warps from different blocks
    transformOneWarp(builder, loc);
  } else if (numWarps == 8) { // pingpong between warps from the same block
    transformTwoWarp(builder, loc);
    builder.setInsertionPointAfter(forOp);
    // barrier before starting pingpong
    auto preBarrier = builder.create<gpu::BarrierOp>(loc);

    // condbarrier::second_half after forOp
    auto constZero = builder.create<arith::ConstantIntOp>(loc, 0, 1);
    auto condBarrierLow =
        builder.create<triton::amdgpu::condBarrierOp>(loc, constZero);

    // condbarrier::first_half after barrier
    preBarrier->moveBefore(forOp);
    builder.setInsertionPointAfter(preBarrier);
    auto constOne = builder.create<arith::ConstantIntOp>(loc, 1, 1);
    auto condBarrierHigh =
        builder.create<triton::amdgpu::condBarrierOp>(loc, constOne);
  }
}

class TritonAMDGPUBlockPingpongPass
    : public TritonAMDGPUBlockPingpongBase<TritonAMDGPUBlockPingpongPass> {
public:
  TritonAMDGPUBlockPingpongPass() = default;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    int32_t numWarps = ttg::TritonGPUDialect::getNumWarps(m);

    m.walk([&](scf::ForOp forOp) {
      Pingponger pingponger(forOp, numWarps);
      pingponger.getDotPingponged();
    });
  }
};

std::unique_ptr<Pass> mlir::createTritonAMDGPUBlockPingpongPass() {
  return std::make_unique<TritonAMDGPUBlockPingpongPass>();
}
