#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define GEN_PASS_CLASSES
#include "TritonAMDGPUTransforms/Passes.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace {

// This pass transforms a for-loop calculating a GEMM. Main purpose of the
// transform is improve the efficiency of the GPU dot instruction (mfma)
// by interleaving the execution of two warps on each SIMD. Especially it groups
// instructions into Dot and Memory clusters so they can efficiently run in
// parallel. Also this pass inserts `rocdl.s.setprio` operation and
// `amdgpu.cond_barrier` to run two parallel warps in synchronization.
// This scheduling doesn't help improving the memory latency itself but it
// relies on software-pipelining to hide the global latency. Likely to improve
// the performance of compute-bound cases.
class Pingponger {
  scf::ForOp forOp;
  SmallVector<tt::LoadOp> gLoadOps;
  SmallVector<ttg::LocalLoadOp> lLoadOps;
  SmallVector<ttg::LocalStoreOp> lStoreOps;
  SmallVector<tt::DotOp> dotOps;
  SmallVector<SmallVector<Operation *>> subViewOps;
  SmallVector<SmallVector<Operation *>> loadSliceOps;
  SmallVector<Operation *> dotSliceOps;
  SmallVector<Value> constOffsets;
  Operation *lastInsertedOp;

  // rocdl.s.setprio will be mapped to `s_setprio` instruction which set the
  // priority of the warp within a SIMD, determines which warp to occupy the
  // instruction unit when they compete on the same instruction.
  // We use this instruction in the pingpong sheduling to prevent warps from
  // entering into the dot cluster while the other warp is still busy in the dot
  // cluster. Otherwise pingpong pattern can be broken and performance drops.
  // Currently pingpong only handles two warps, we only need 0/1 priorities.
  int lowPriority = 0;
  int highPriority = 1;
  int32_t kWidth;
  int32_t numWarps;

public:
  Pingponger(scf::ForOp forOp, int32_t numWarps)
      : forOp(forOp), numWarps(numWarps) {}
  void getDotPingponged();

private:
  void genOffsetConstants(Location loc, OpBuilder &builder, unsigned numSlices,
                          int64_t sliceWidth);
  LogicalResult genLocalSlice(OpBuilder &builder, Value v,
                              Attribute dotEncoding, unsigned opIdx,
                              unsigned numSlices, int64_t sliceWidth);
  void transformOnePPClusters(OpBuilder &builder, Location loc);
  LogicalResult transformFourPPClusters(OpBuilder &builder, Location loc);
  void addAsymmetricSyncToLoop(OpBuilder &builder, Location loc);
  void updateOpInsertion(Operation *Op);
  void appendOp(Operation *Op);
  void appendSlicedLoadAB(int slice);
  void appendClusterBarrier(OpBuilder &builder, Location loc);
  void appendOpWithPrio(OpBuilder &builder, Operation *Op, Location loc);
};

void Pingponger::updateOpInsertion(Operation *op) { lastInsertedOp = op; }
void Pingponger::appendOp(Operation *op) {
  assert(lastInsertedOp != nullptr);
  op->moveAfter(lastInsertedOp);
  lastInsertedOp = op;
}
void Pingponger::appendSlicedLoadAB(int slice) {
  appendOp(subViewOps[0][slice]);
  appendOp(loadSliceOps[0][slice]);
  appendOp(subViewOps[1][slice]);
  appendOp(loadSliceOps[1][slice]);
}
// Asymmetrically synchronized loop in the pingpong scheduling synchronizes all
// the warps at the end of each instruction cluster. Since cond_barrier
// triggered a barrier for only half of the warps in a block, at the point
// this clusterBarrier is called, half warps are at dot cluster and the others
// are at the memory cluster.
// Also, SchedBarrier with `0` is set here to tell compiler backend not to
// reorder any instruction across this point.
void Pingponger::appendClusterBarrier(OpBuilder &builder, Location loc) {
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  // MembarAnalysis can recognize gpu::BarrierOp and skip inserting additional
  // barrier
  appendOp(builder.create<gpu::BarrierOp>(loc));
}
void Pingponger::appendOpWithPrio(OpBuilder &builder, Operation *op,
                                  Location loc) {
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, highPriority));
  appendOp(op);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, lowPriority));
}

// Transform a loop into one Dot - Memory (ping - pong) clusters
// Each cluster, especially the Dot cluster is guarded with setprio(1->0) so
// each warp can complete the execution of the cluster without being
// interrupted. This is also supposed to be used with the numWarps=4 case where
// each SIMD runs two warps from different blocks and those two warps don't need
// to be synchronized together.
// Splitting loading A/B and interleave global/local load in order to prevent
// the stalls.
// sched.barriers with 0 mask were used to enforce the boundary of the
// high-level operations, inserting `setPrio` also has a same effect of
// instruction scheduling boundary, too.
void Pingponger::transformOnePPClusters(OpBuilder &builder, Location loc) {
  // Memory cluster #0
  updateOpInsertion(lLoadOps[0]);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, highPriority));
  appendOp(gLoadOps[0]);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));
  appendOp(lLoadOps[1]);
  appendOp(builder.create<ROCDL::SetPrioOp>(loc, lowPriority));
  appendOp(gLoadOps[1]);
  appendOp(builder.create<ROCDL::SchedBarrier>(loc, 0));

  // Dot cluster #0
  appendOpWithPrio(builder, dotOps[0], loc);
}

void Pingponger::genOffsetConstants(Location loc, OpBuilder &builder,
                                    unsigned numSlices, int64_t sliceWidth) {
  for (int i = 0; i < numSlices; i++) {
    int64_t offset = sliceWidth * i;
    constOffsets.push_back(
        builder.create<arith::ConstantIntOp>(loc, offset, 32));
  }
}

// Splits given local_loads for dot into multiple subviews and local_loads. This
// function tries to slice the local_load into the given number of the slices,
// generates ops when succeed, return fail() otherwise.
LogicalResult Pingponger::genLocalSlice(OpBuilder &builder, Value v,
                                        Attribute dotEncoding, unsigned opIdx,
                                        unsigned numSlices,
                                        int64_t sliceWidth) {
  SmallVector<Operation *> slices;
  SmallVector<Operation *> subviews;
  auto memDesc = v.getDefiningOp()->getOperand(0);
  auto type = cast<ttg::MemDescType>(memDesc.getType());
  auto encoding = cast<RankedTensorType>(v.getType()).getEncoding();
  auto srcEncoding = cast<ttg::DotOperandEncodingAttr>(encoding);
  SmallVector<int64_t> shape = llvm::to_vector(type.getShape());
  Type elementType = type.getElementType();
  int64_t kIdx = opIdx == 0 ? 1 : 0;
  shape[kIdx] = sliceWidth;
  // Each slice cannot be smaller than the smallest supported mfma width.
  if (sliceWidth < 16)
    return failure();
  auto dotOperandEnc = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, srcEncoding.getKWidth());
  auto subviewDescType = ttg::MemDescType::get(
      shape, elementType, type.getEncoding(), type.getMemorySpace());
  for (int i = 0; i < numSlices; i++) {
    SmallVector<Value> offsetsVal;
    SmallVector<int64_t> offsets = {0, 0};
    offsets[kIdx] = i;
    for (int64_t off : offsets) {
      offsetsVal.push_back(constOffsets[off]);
    }
    Value newSmem = builder.create<ttg::MemDescSubviewOp>(
        v.getLoc(), subviewDescType, memDesc, offsetsVal);
    Value prefetchSlice = builder.create<ttg::LocalLoadOp>(
        v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
        newSmem);
    subviews.push_back(newSmem.getDefiningOp());
    slices.push_back(prefetchSlice.getDefiningOp());
  }
  subViewOps.push_back(subviews);
  loadSliceOps.push_back(slices);
  return success();
}

// Transform a loop into four Dot - Memory (ping - pong) clusters
// This transfrom is useful when the original dot tile is too large that there's
// no enough register to hold data for a Dot cluster. This path slices the dot
// into four pieces and pair with four clusters of reordered memory operations.
// There are multiple guards at the boundary of each cluster.
// (1) sched.barrier : with mask0 to prevent compiler backed from reroder
//  instructions across the boundary
// (2) gpu.barrier : ensures asymmetric synchronization at each point
// (3) setprio (1->0) : in order to avoid incomming warp overtaking resource
//  while the other warp is actively using it.
//
// Here's overview of the instruction clusters
// mem0: global load A, local load A(1/4), local load B(1/4)
// dot0: dot A(1/4) * B(1/4)
// mem1: global load B, local load A(2/4), local load B(2/4)
// dot1: dot A(2/4) * B(2/4)
// mem2: local load A(3/4, 4/4), local load B(3/4, 4/4)
// dot2: dot A(3/4) * B(3/4)
// mem3: local store A and B
// dot3: dot A(4/4) * B(4/4)

LogicalResult Pingponger::transformFourPPClusters(OpBuilder &builder,
                                                  Location loc) {
  // First, slice local_loads and dot into 4 parts
  unsigned numSlices = 4;
  auto op = cast<tt::DotOp>(dotOps[0]);
  builder.setInsertionPointToStart(forOp.getBody());
  auto typeB = op.getB().getType();
  auto shapeB = typeB.getShape();
  int64_t sliceWidth = shapeB[0] / numSlices;
  if (shapeB[0] % numSlices != 0)
    return failure();
  genOffsetConstants(loc, builder, numSlices, sliceWidth);
  builder.setInsertionPointAfter(gLoadOps[0]);
  auto dotEncoding = op.getType().getEncoding();
  if (genLocalSlice(builder, op.getA(), dotEncoding, 0, numSlices, sliceWidth)
          .failed() ||
      genLocalSlice(builder, op.getB(), dotEncoding, 1, numSlices, sliceWidth)
          .failed())
    return failure();

  // Clone dots four times to consume the slices
  Operation *prevDot = op;
  for (int i = 0; i < numSlices; i++) {
    IRMapping mapping;
    mapping.map(op.getA(), loadSliceOps[0][i]->getResult(0));
    mapping.map(op.getB(), loadSliceOps[1][i]->getResult(0));
    if (i > 0)
      mapping.map(op.getC(), prevDot->getResult(0));
    auto newOp = builder.clone(*op, mapping);
    prevDot = newOp;
    dotSliceOps.push_back(newOp);
  }
  op->replaceAllUsesWith(prevDot);
  op->erase();
  for (auto loads : lLoadOps)
    loads->erase();

  builder.setInsertionPointAfter(gLoadOps[1]);
  // Reorder operations into four mem/dot clusters

  // mem0: global load A, local load A(1/4), local load B(1/4)
  updateOpInsertion(gLoadOps[1]);
  appendSlicedLoadAB(/*slice=*/0);
  appendClusterBarrier(builder, loc);

  // dot0 (1/4)
  appendOpWithPrio(builder, dotSliceOps[0], loc);
  appendClusterBarrier(builder, loc);

  // mem1: global load B, local load A(2/4), local load B(2/4)
  appendOp(gLoadOps[1]);
  appendSlicedLoadAB(/*slice=*/1);
  appendClusterBarrier(builder, loc);

  // dot1 (2/4)
  appendOpWithPrio(builder, dotSliceOps[1], loc);
  appendClusterBarrier(builder, loc);

  // mem2: local load A(3/4, 4/4), local load B(3/4, 4/4)
  appendSlicedLoadAB(/*slice=*/2);
  appendSlicedLoadAB(/*slice=*/3);
  appendClusterBarrier(builder, loc);

  // dot2 (3/4)
  appendOpWithPrio(builder, dotSliceOps[2], loc);
  appendClusterBarrier(builder, loc);

  // mem3: local store A and B
  updateOpInsertion(lStoreOps[1]);
  appendClusterBarrier(builder, loc);

  // dot3 (4/4)
  appendOpWithPrio(builder, dotSliceOps[3], loc);
  appendClusterBarrier(builder, loc);

  return success();
}

// This function wraps forOp with cond_barrier. First, hold half of the warps
// (warpHigh) in a block before the loop so the barriers in the loop synchronize
// warps at the different point per the warp groups. After the loop, hold
// proceeding warps (warpLow) by calling cond_barrier on them.
void Pingponger::addAsymmetricSyncToLoop(OpBuilder &builder, Location loc) {
  builder.setInsertionPointAfter(forOp);
  // Set barrier before starting the loop. This resolves any remaining required
  // synchronization before beginning the specialized asymmetric
  // synchronization.
  auto preBarrier = builder.create<gpu::BarrierOp>(loc);
  preBarrier->moveBefore(forOp);
  builder.setInsertionPointAfter(preBarrier);

  // Insert condbarrier::second_half before starting the loop
  auto i32ty = builder.getIntegerType(32);
  auto workIDX = builder.create<ROCDL::ThreadIdXOp>(loc, i32ty);
  auto constZero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  auto constWarpSize = builder.create<arith::ConstantIntOp>(loc, 256, 32);
  auto warpIDX = builder.create<arith::DivSIOp>(loc, workIDX, constWarpSize);
  auto warpLow = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               warpIDX, constZero);
  auto warpHigh = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                warpIDX, constZero);
  auto condBarrierHigh =
      builder.create<tt::amdgpu::CondBarrierOp>(loc, warpHigh);

  // Insert condbarrier::first_half after the end of the loop
  builder.setInsertionPointAfter(forOp);
  auto condBarrierLow = builder.create<tt::amdgpu::CondBarrierOp>(loc, warpLow);
}

void Pingponger::getDotPingponged() {
  OpBuilder builder(forOp);
  MLIRContext *ctx = forOp.getContext();
  Location loc = forOp.getLoc();
  auto f16ty = builder.getF16Type();

  forOp->walk([&](Operation *op) {
    if (auto gLoad = dyn_cast<tt::LoadOp>(op))
      gLoadOps.push_back(gLoad);
    else if (auto lLoad = dyn_cast<ttg::LocalLoadOp>(op)) {
      // This scheduling doesn't help hiding intra-warp latency. So, we only
      // collect local_load ops that are software pipelined, which means their
      // source is from loop carried values
      auto src = lLoad.getSrc();
      if (auto arg = mlir::dyn_cast<BlockArgument>(src))
        if (auto tiedLoopInit = forOp.getTiedLoopInit(arg))
          if (tiedLoopInit->get())
            lLoadOps.push_back(lLoad);
    } else if (auto lStore = dyn_cast<ttg::LocalStoreOp>(op))
      lStoreOps.push_back(lStore);
    else if (auto pingpongDot = dyn_cast<tt::DotOp>(op))
      if (pingpongDot.getType().getRank() == 2)
        dotOps.push_back(pingpongDot);
  });

  // Currently, pingpong scheduling is known as helpful under limited condition.
  // Individual conditions are checked while collecting each operation such as
  // software pipelining and dot rank=2. Also only accept the for-loop with
  // supported combination of operations because this transformation is very
  // tightly scheduling the latencies.
  if (gLoadOps.size() != 2 || lLoadOps.size() != 2 || dotOps.size() != 1)
    return;

  // Pingpong scheduling tries to form two different types of the instruction
  // clusters, i.e., Dot clusters and Memory clusters. While each SIMD has
  // two concurrent warps, both warps can execute a different type of
  // instruction cluster in parallel.Here are currently available patterns,
  // more patterns could be added later.
  //
  // (1) One Dot-Memory (ping-pong) cluster
  //  :Ideal to support small tile size e.g., 128x128x64_FP16. Where amount
  //   of the data used per each iteration is small enough and not causing
  //   local_load waiting or register spilling. Currently used for numWarps=4
  //   case where SIMD can hold two warps from different blocks.
  //
  // (2) Four Dot-Memory (ping-pongx4) clusters
  //  :Useful for the larger tile size e.g., 256x256x64_FP16. Clustering
  //   the Dot instruction (mfma) all together without fetching data requires
  //   GPU to hold all the data for the calculation. Such large tile size
  //   exceeds the amount of register GPU has so, we need to split the dot
  //   into several pieces.

  // TODO:
  // - Add transformTwoPPClusters for the medium size tiles
  // - Add definition of small/medium/large tile size considering data-type
  //   so we can choose the transfrom per given tile size.
  if (numWarps == 4) { // pingpong between warps from different blocks
    // transfor a loop with small tile size
    transformOnePPClusters(builder, loc);
  } else if (numWarps == 8) { // pingpong between warps from the same block
    // transfor a loop with large tile size which requires dots to be sliced
    if (transformFourPPClusters(builder, dotOps[0]->getLoc()).failed())
      return;

    // Let half of the warps start the loop first and the others follow later
    // but in the synchronized way. This can be accompished by calling
    // cond_barrier for the second half before the beginning of the loop so they
    // can wait until the first half hit the first barrier in the loop. Also
    // need to call cond_barrier for the first_half after exiting the loop, so
    // all warps can converge again.
    addAsymmetricSyncToLoop(builder, loc);
  }
}

class TritonAMDGPUBlockPingpongPass
    : public TritonAMDGPUBlockPingpongBase<TritonAMDGPUBlockPingpongPass> {
public:
  TritonAMDGPUBlockPingpongPass() = default;
  void runOnOperation() override {
    ModuleOp m = getOperation();
    int32_t numWarps = ttg::TritonGPUDialect::getNumWarps(m);
    for (auto funcOp : m.getOps<tt::FuncOp>()) {
      funcOp.walk([&](scf::ForOp forOp) {
        Pingponger pingponger(forOp, numWarps);
        pingponger.getDotPingponged();
      });
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createTritonAMDGPUBlockPingpongPass() {
  return std::make_unique<TritonAMDGPUBlockPingpongPass>();
}
