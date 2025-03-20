#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include <unordered_set>

#define DEBUG_TYPE "triton-ping-pong-sync"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttng = ::mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

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

static Value createGetAsyncTaskId(OpBuilder &builder, Operation *op) {
  auto loc = op->getLoc();
  return builder.create<ttng::GetAsyncTaskIdOp>(loc);
}

static bool isInnermostLoop(scf::ForOp forOp) {
  for (Operation &nestedOp : forOp.getBody()->getOperations()) {
    if (isa<scf::ForOp>(nestedOp)) {
      return false;
    }
  }
  return true;
}

#define GEN_PASS_DEF_TRITONGPUPINGPONGSYNC
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUPingPongSyncPass
    : public impl::TritonGPUPingPongSyncBase<TritonGPUPingPongSyncPass> {
public:
  using impl::TritonGPUPingPongSyncBase<
      TritonGPUPingPongSyncPass>::TritonGPUPingPongSyncBase;

  enum class ResourceType {
    Gemm,
    OtherComp,
  };
  const int PING_BARRIER = 9;
  const int PONG_BARRIER = 10;

  unsigned getLoopDepth(Operation *op) {
    unsigned depth = 0;
    auto pOp = op->getParentOfType<scf::ForOp>();
    while (pOp) {
      ++depth;
      pOp = pOp->getParentOfType<scf::ForOp>();
    }
    return depth;
  }

  void
  getNestedFor(scf::IfOp ifOp,
               DenseMap<unsigned, SmallVector<Operation *>> &loopDepthMap) {
    ifOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
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
  void runOnFuncOp(triton::FuncOp funcOp) {
    // Insert sync points in ForOp for consumer warp groups. Enable this pass
    // when number of consumer warp groups == 2.
    if (numConsumerGroups != 2)
      return;
    if (!mlir::triton::tools::getBoolEnv("ENABLE_PINGPONG"))
      return;

    SmallVector<scf::ForOp> loops;
    // Identify ForOps for consumer warp groups. Here we assume taskId 0 is for
    // producer. This pass handles the case of a single forOp for two consumer
    // warp groups.
    // Find top-most IfOps, and find the top level ForOp, assuming only one top
    // level ForOp. A few use cases: 1> Persistent with ForOp containing both
    // cuda and tensor 2> Persistent with ForOp containing tensor, then epilogue
    // with cuda
    DenseMap<unsigned, SmallVector<Operation *>> loopDepthMap1;
    DenseMap<unsigned, SmallVector<Operation *>> loopDepthMap2;
    for (auto &block : funcOp.getBody().getBlocks()) {
      for (Operation &bodyOp : block.getOperations()) {
        Operation *op = &bodyOp;
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          int wgId = getSingleTaskId(op);
          // Assume taskId 0 is for producer. Assume we will visit taskId 1
          // first.
          if (wgId == 1) {
            getNestedFor(ifOp, loopDepthMap1);
          } else if (wgId == 2) {
            getNestedFor(ifOp, loopDepthMap2);
          }
        }
      }
    }
    // Verify loopDepthMap1 and loopDepthMap2: a single ForOp at depth of 0
    // and a single ForOp at depth of 1.
    LDBG("Found loops: " << loopDepthMap1.size());
    if (loopDepthMap1.empty())
      return;
    if (loopDepthMap1[0].size() != 1)
      return;
    bool hasPersistent = loopDepthMap1.find(1) != loopDepthMap1.end();

    // Assume two loops have the same ops. Check innermost loop.
    SmallVector<Operation *> starts, ends;
    for (unsigned iter = 0; iter < 2; ++iter) {
      Operation *op = iter == 0 ? loopDepthMap1[hasPersistent ? 1 : 0][0]
                                : loopDepthMap2[hasPersistent ? 1 : 0][0];
      auto forOp = dyn_cast<scf::ForOp>(op);
      Operation *startOfGemm = nullptr;
      Operation *endOfGemm = nullptr;
      // A simple heuristic for now:
      //   Mark the start of a gemm section when hitting a DotLike op.
      //   Mark the end of a gemm section once hitting an expensive non-dot
      //   computation op.
      for (auto &op : forOp.getBody()->without_terminator()) {
        if (startOfGemm && endOfGemm)
          break;
        bool hasDot, isCudaCore;
        bool hasError = false;
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          // if containing expensive cuda core op
          hasError = categorizeIf(ifOp, hasDot, isCudaCore);
        } else {
          LLVM_DEBUG({
            LDBG("walk for");
            op.dump();
          });
          hasDot = isa<mlir::triton::DotOpInterface>(op);
          // hasDot = isa<nvidia_gpu::WarpGroupDotOp>(&op);
          isCudaCore = isExpensiveComp(&op);
          LDBG("walk for " << hasDot << " " << isCudaCore);
        }
        if (hasError || (hasDot && isCudaCore))
          break;
        if (hasDot && !isCudaCore && startOfGemm == nullptr) {
          startOfGemm = &op;
          continue;
        }
        if (!hasDot && isCudaCore && startOfGemm) {
          endOfGemm = &op;
          break;
        }
      }
      if (startOfGemm) {
        LLVM_DEBUG({
          LDBG("found start of tensor core ops");
          startOfGemm->dump();
        });
      }
      if (endOfGemm) {
        LLVM_DEBUG({
          LDBG("found end of tensor core ops");
          endOfGemm->dump();
        });
      }

      if (!startOfGemm || !endOfGemm)
        return;
      starts.push_back(startOfGemm);
      ends.push_back(endOfGemm);
    }
    // TODO: epilogue overlapping.
    {
      // "bar.arrive 9, 256" only when task Id is 2.
      Operation *outerLoopTask2 = loopDepthMap2[0][0];
      OpBuilder builder(outerLoopTask2);
      builder.setInsertionPoint(outerLoopTask2);
      auto forLoc = outerLoopTask2->getLoc();
      Value pingBarrier =
          builder.create<arith::ConstantIntOp>(forLoc, PING_BARRIER, 32);
      Value numThreads = builder.create<arith::ConstantIntOp>(forLoc, 256, 32);
      builder.create<ttng::NamedBarrierArriveOp>(forLoc, pingBarrier,
                                                 numThreads);
    }
    for (unsigned idx = 0; idx < 2; ++idx) {
      Operation *op = idx == 0 ? loopDepthMap1[hasPersistent ? 1 : 0][0]
                               : loopDepthMap2[hasPersistent ? 1 : 0][0];
      auto forOp = dyn_cast<scf::ForOp>(op);
      OpBuilder builder(forOp);
      Operation *startOfGemm = starts[idx];
      Operation *endOfGemm = ends[idx];

      // FIXME: hard-code using named barrier 9 and 10 in this pass.
      // Prior to the forOp, add "bar.arrive 9, 256" only when task Id is 2.
      // At startOfGemm, insert "bar.sync 8+taskId, 256"
      // At endOfGemm, insert "bar.arrive 11-taskId, 256"
      builder.setInsertionPoint(forOp);
      auto forLoc = forOp->getLoc();

      // FIXME: hard-code total number of threads to be 256 when
      // numConsumerGroups is 2.
      Value numThreads = builder.create<arith::ConstantIntOp>(forLoc, 256, 32);
      // for taskId of 1, generate: bar.sync pingBarrier; bar.arrive pongBarrier
      // for taskId of 2, outside of the loop, generate bar.arrive pingBarrier
      //   inside the loop, generate bar.sync pongBarrier; bar.arrive
      //   pingBarrier
      Value pingBarrier =
          builder.create<arith::ConstantIntOp>(forLoc, PING_BARRIER, 32);

      int wgId = getSingleTaskId(forOp);
      // At startOfGemm, insert "bar.sync 9 or 10, 256"
      builder.setInsertionPoint(startOfGemm);
      auto loc = startOfGemm->getLoc();
      Value syncBarrier = builder.create<arith::ConstantIntOp>(
          loc, wgId == 1 ? PING_BARRIER : PONG_BARRIER, 32);
      builder.create<ttng::NamedBarrierWaitOp>(loc, syncBarrier, numThreads);

      // At endOfGemm, insert "bar.arrive 10 or 9, 256"
      Operation *insertBefore = endOfGemm;
      insertBefore = moveBackward(endOfGemm, forOp);
      builder.setInsertionPoint(insertBefore);
      auto loc2 = endOfGemm->getLoc();
      Value arriveBarrier = builder.create<arith::ConstantIntOp>(
          loc2, wgId == 1 ? PONG_BARRIER : PING_BARRIER, 32);
      builder.create<ttng::NamedBarrierArriveOp>(loc2, arriveBarrier,
                                                 numThreads);
    }
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
    LLVM_DEBUG({
      LDBG("post pass");
      getOperation()->dump();
    });
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
