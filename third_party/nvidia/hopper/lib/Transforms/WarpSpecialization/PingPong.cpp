#include "Utility.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "nvidia/hopper/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <unordered_set>

#define DEBUG_TYPE "nvgpu-ping-pong-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {

// Returns the taskId if op has a single taskId, otherwise, returns -1.
static int getSingleTaskId(Operation *op) {
  auto asyncTasks = getAsyncTaskIds(op);
  if (asyncTasks.size() != 1)
    return -1;
  return asyncTasks[0];
}

// Treat exp2, mulf, addf, reduce as expensive computation when data type is
// a tensor type of 1D or higher.
static bool isExpensiveComp(Operation *op) {
  if (!isa<arith::MulFOp>(op) && !isa<math::Exp2Op>(op) &&
      !isa<arith::AddFOp>(op) && !isa<mlir::triton::ReduceOp>(op))
    return false;
  auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  return tensorTy && tensorTy.getRank() >= 1;
}

static bool isInnermostLoop(scf::ForOp forOp) {
  for (Operation &nestedOp : forOp.getBody()->getOperations()) {
    if (isa<scf::ForOp>(nestedOp)) {
      return false;
    }
  }
  return true;
}

// Ideally we should have GEMM, SFU, and OtherComp
enum class ResourceType {
  Gemm,
  OtherComp,
};
// FIXME: hard-coded named barriers.
const int PING_BARRIER = 9;
const int PONG_BARRIER = 10;

static unsigned getLoopDepth(Operation *op) {
  unsigned depth = 0;
  auto pOp = op->getParentOfType<scf::ForOp>();
  while (pOp) {
    ++depth;
    pOp = pOp->getParentOfType<scf::ForOp>();
  }
  return depth;
}

void getNestedFor(Region *partition,
                  DenseMap<unsigned, SmallVector<Operation *>> &loopDepthMap) {
  partition->walk([&](Operation *subOp) {
    if (dyn_cast<scf::ForOp>(subOp)) {
      unsigned tDepth = getLoopDepth(subOp);
      loopDepthMap[tDepth].push_back(subOp);
    }
  });
}

Operation *moveBackward(Operation *endofGemm, scf::ForOp forOp) {
  SmallVector<Operation *> opList;
  for (auto &op : forOp.getBody()->without_terminator()) {
    opList.push_back(&op);
  }
  bool found = false;
  Operation *newEnd = endofGemm;
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    Operation *op = *it;
    if (op == endofGemm) {
      found = true;
      continue;
    }
    if (found && isa<mlir::triton::DotOpInterface>(op)) {
      break;
    }
    if (found)
      newEnd = op;
  }
  return newEnd;
}

bool categorizeIf(scf::IfOp ifOp, bool &hasDot, bool &hasExpCudaOp) {
  hasDot = false;
  hasExpCudaOp = false;
  bool hasFor = false;
  ifOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    LLVM_DEBUG({
      LDBG("walk if");
      subOp->dump();
    });
    if (isa<scf::ForOp>(subOp)) {
      hasFor = true;
    } else if (isa<mlir::triton::DotOpInterface>(subOp)) {
      hasDot = true;
    } else if (isExpensiveComp(subOp)) {
      hasExpCudaOp = true;
    }
    LDBG("---- " << hasDot << " " << hasExpCudaOp << " " << hasFor);
  });
  LDBG("after walk if " << hasDot << " " << hasExpCudaOp << " " << hasFor);
  return hasFor;
}

static void handleWarpSpec(ttg::WarpSpecializeOp wsOp) {
  // ForOps in partition 0, map from loop depth to the ForOp.
  DenseMap<unsigned, SmallVector<Operation *>> loopDepthPartition0;
  // ForOps in partition 1, map from loop depth to the ForOp.
  DenseMap<unsigned, SmallVector<Operation *>> loopDepthPartition1;
  unsigned partitionId = 0;
  for (Region *region : wsOp.getPartitionRegions()) {
    LDBG("-- region " << region->getNumArguments());
    // Assume default for producer.
    if (partitionId == 0) {
      getNestedFor(region, loopDepthPartition0);
    } else if (partitionId == 1) {
      getNestedFor(region, loopDepthPartition1);
    }
    ++partitionId;
  }

  // Verify a single ForOp at depth of 0, i.e a single outer ForOp.
  LDBG("Found loops: " << loopDepthPartition0.size());
  if (loopDepthPartition0.empty() || loopDepthPartition0[0].size() != 1)
    return;
  // Have a ForOp at depth of 1.
  bool hasPersistent = loopDepthPartition0.find(1) != loopDepthPartition0.end();

  // Find the region of gemms via [startOfGemm, endOfGemm)
  SmallVector<Operation *> starts, ends;
  for (unsigned iter = 0; iter < 2; ++iter) {
    // Check partition 0 first, then partition 1.
    // Find the innermost ForOp (i.e depth of 1 for persistent).
    Operation *op = iter == 0 ? loopDepthPartition0[hasPersistent ? 1 : 0][0]
                              : loopDepthPartition1[hasPersistent ? 1 : 0][0];
    auto forOp = dyn_cast<scf::ForOp>(op);
    // A simple heuristic for now:
    //   Mark the start of a gemm section when hitting a DotLike op.
    //   Mark the end of a gemm section once hitting an expensive non-dot
    //   computation op.
    Operation *startOfGemm = nullptr;
    Operation *endOfGemm = nullptr;
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (startOfGemm && endOfGemm)
        break;
      bool hasDot, isCudaCore;
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        // Categorize IfOp to see if it contains dot or expensive cuda core op.
        bool hasError = categorizeIf(ifOp, hasDot, isCudaCore);
        // Bail out if IfOp contains both Dot and cuda core ops or contains
        // ForOp.
        if (hasError || (hasDot && isCudaCore))
          break;
      } else {
        hasDot = isa<mlir::triton::DotOpInterface>(op);
        isCudaCore = isExpensiveComp(&op);
        if (hasDot || isCudaCore) {
          LLVM_DEBUG({
            LDBG("walk op in for");
            op.dump();
          });
          LDBG("-- hasDot " << hasDot << " cudaCoreOp " << isCudaCore);
        }
      }
      if (hasDot && startOfGemm == nullptr) {
        // This is a dotop and we are trying to set startOfGemm.
        startOfGemm = &op;
        LLVM_DEBUG({
          LDBG("set start of Gemm region");
          startOfGemm->dump();
        });
        continue;
      }
      if (isCudaCore && startOfGemm) {
        // Already found startOfGemm, set endOfGemm prior to an expensive cuda
        // core op.
        endOfGemm = &op;
        LLVM_DEBUG({
          LDBG("set end of Gemm region");
          endOfGemm->dump();
        });
        break;
      }
    }
    if (!startOfGemm || !endOfGemm)
      return;
    starts.push_back(startOfGemm);
    ends.push_back(endOfGemm);
  }

  // TODO: epilogue overlapping.
  // "bar.arrive PING, 256" prior to outer loop for partition 1.
  Operation *outerLoopPartition1 = loopDepthPartition1[0][0];
  OpBuilder builder(outerLoopPartition1);
  builder.setInsertionPoint(outerLoopPartition1);
  auto forLoc = outerLoopPartition1->getLoc();
  Value pingBarrier =
      builder.create<arith::ConstantIntOp>(forLoc, PING_BARRIER, 32);
  // 256 threads for one partition of 4 warps.
  Value numThreads = builder.create<arith::ConstantIntOp>(forLoc, 256, 32);
  builder.create<ttng::NamedBarrierArriveOp>(forLoc, pingBarrier, numThreads);

  for (unsigned idx = 0; idx < 2; ++idx) {
    // Find the innermost ForOp (i.e depth of 1 for persistent).
    Operation *op = idx == 0 ? loopDepthPartition0[hasPersistent ? 1 : 0][0]
                             : loopDepthPartition1[hasPersistent ? 1 : 0][0];
    auto forOp = dyn_cast<scf::ForOp>(op);
    OpBuilder builder(forOp);
    Operation *startOfGemm = starts[idx];
    Operation *endOfGemm = ends[idx];

    // At startOfGemm, insert "bar.sync PING, 256" for partition 0 or "bar.sync
    // PONG" for partition 1 At endOfGemm, insert "bar.arrive PONG, 256" for
    // partition 0 or "bar.arrive PING" for partition 1
    builder.setInsertionPoint(forOp);
    auto forLoc = forOp->getLoc();
    // Hard-code number of threads to be 256 for each partition.
    Value numThreads = builder.create<arith::ConstantIntOp>(forLoc, 256, 32);

    builder.setInsertionPoint(startOfGemm);
    auto loc = startOfGemm->getLoc();
    Value syncBarrier = builder.create<arith::ConstantIntOp>(
        loc, idx == 0 ? PING_BARRIER : PONG_BARRIER, 32);
    builder.create<ttng::NamedBarrierWaitOp>(loc, syncBarrier, numThreads);

    Operation *insertBefore = moveBackward(endOfGemm, forOp);
    builder.setInsertionPoint(insertBefore);
    auto loc2 = endOfGemm->getLoc();
    Value arriveBarrier = builder.create<arith::ConstantIntOp>(
        loc2, idx == 0 ? PONG_BARRIER : PING_BARRIER, 32);
    builder.create<ttng::NamedBarrierArriveOp>(loc2, arriveBarrier, numThreads);
  }
}

void doPingPongSync(triton::FuncOp &funcOp, unsigned numWarpGroups) {
  // Insert sync points in ForOp for consumer warp groups. Enable this pass
  // when number of consumer warp groups == 2.
  if (numWarpGroups != 3)
    return;

  SmallVector<scf::ForOp> loops;
  // Identify ForOps for consumer warp groups. Check partitions to find regions
  // of gemms.
  DenseMap<unsigned, SmallVector<Operation *>> loopDepthPartition0;
  DenseMap<unsigned, SmallVector<Operation *>> loopDepthPartition1;
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &bodyOp : block.getOperations()) {
      Operation *op = &bodyOp;
      if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(op)) {
        handleWarpSpec(wsOp);
      }
    }
  }
}

#define GEN_PASS_DEF_NVGPUTESTPINGPONGSYNC
#include "nvidia/hopper/include/Transforms/Passes.h.inc"

class NVGPUTestPingPongSyncPass
    : public impl::NVGPUTestPingPongSyncBase<NVGPUTestPingPongSyncPass> {
public:
  using impl::NVGPUTestPingPongSyncBase<
      NVGPUTestPingPongSyncPass>::NVGPUTestPingPongSyncBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numWarpGroups >= 3)
      // Assuming one producer warp group.
      doPingPongSync(funcOp, numWarpGroups);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace mlir
