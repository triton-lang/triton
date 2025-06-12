#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/MMAv5PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu {
namespace {

//===----------------------------------------------------------------------===//
// assignLatencies
//===----------------------------------------------------------------------===//

// Return true if the preconditions for pipelining the loop are met.
bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (loopHasDistGreaterThanOne(forOp))
    return false;
  // Don't pipeline outer loops.
  if (isOuterLoop(forOp))
    return false;
  return true;
}

bool hasLatenciesAssigned(scf::ForOp forOp) {
  auto helper = TritonDialect::getLoaded(forOp)->getLatencyAttrHelper();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (helper.getAttr(&op))
      return true;
  }
  return false;
}

void assignUserProvidedLatencies(scf::ForOp forOp,
                                 DenseMap<Operation *, int> &opLatency) {
  auto helper = TritonDialect::getLoaded(forOp)->getLatencyAttrHelper();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto latencyAttr = helper.getAttr(&op)) {
      opLatency[&op] = latencyAttr.getInt();
    }
  }
}

class AssignLoadLatencies {
public:
  AssignLoadLatencies(scf::ForOp forOp, int numStages,
                      DenseMap<Operation *, int> &opLatency)
      : forOp(forOp), numStages(numStages), opLatency(opLatency) {};

  void run() {
    bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
    ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
    tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    llvm::MapVector<Operation *, int> loadOpToIndLevel =
        loadOpsToIndirectionLevel(forOp, pipelineWithoutDot, axisInfoAnalysis);
    if (loadOpToIndLevel.empty())
      return;

    // We assume loads with different dist are assigned to different stages.
    // If numStages is 2, we will have no stage available for indirect loads
    // with dist >= 1. In general, when dist is equal to numStages - 1, we
    // should not pipeline it.
    for (auto iter = loadOpToIndLevel.begin();
         iter != loadOpToIndLevel.end();) {
      if (iter->second >= numStages - 1)
        iter = loadOpToIndLevel.erase(iter);
      else
        ++iter;
    }

    // Calculate the stage distance between applicable loads.
    auto vals = llvm::make_second_range(loadOpToIndLevel);
    int maxIndirectionLevel = vals.empty() ? 0 : *llvm::max_element(vals);
    unsigned loadLatency = (numStages - 1) / (maxIndirectionLevel + 1);

    for (auto [loadOp, dist] : loadOpToIndLevel) {
      opLatency[loadOp] = loadLatency;
    }
  }

private:
  scf::ForOp forOp;
  int numStages;
  DenseMap<Operation *, int> &opLatency;

  bool canHaveSharedEncoding(tt::LoadOp op) {
    // If used by an user with DotOp encoding, all the uses must be compatible.
    bool incompatible = false;
    getSharedEncIfAllUsersAreDotEnc(op.getResult(), incompatible);
    return !incompatible;
  }

  bool isPipeliningBeneficial(Operation *op, Operation *finalUser,
                              tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      if (!canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis)) {
        LDBG("Load " << *loadOp << " is too small for pipelining");
        return false;
      }
    }
    if (isa<tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
      return true;
    if (!canHaveSharedEncoding(cast<tt::LoadOp>(op))) {
      LDBG("Load " << *op << " cannot have shared encoding");
      return false;
    }

    ttg::SharedEncodingTrait localAllocEnc;
    if (llvm::any_of(op->getUsers(), [&](Operation *user) {
          return isa<ttg::LocalAllocOp>(user);
        })) {
      for (auto user : op->getUsers()) {
        auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
        if (!localAlloc)
          continue;
        auto enc = mlir::cast<ttg::SharedEncodingTrait>(
            localAlloc.getType().getEncoding());
        if (!localAllocEnc) {
          localAllocEnc = enc;
        }
        if (enc != localAllocEnc) {
          // If the load is used by a LocalAllocOp, all the users need to have
          // the same encoding.
          return false;
        }
      }
    }

    if (localAllocEnc) {
      auto registerTy = cast<RankedTensorType>(op->getResultTypes()[0]);
      auto vecBytes = getCopyVecBytes(registerTy, localAllocEnc);
      if (vecBytes < 4) {
        // At least 4 bytes need to be consecutive for cp.async
        return false;
      }
    }

    return true;
  }

  // Create a map from load ops to their indirection level and the
  // final use of the load op (another load op, or a dot op).
  // Indirection level is "0" for the load op directly used by the dot op,
  // "1" for the load op used by the load op used by the dot op, and so on.
  llvm::MapVector<Operation *, int>
  loadOpsToIndirectionLevel(scf::ForOp forOp, bool pipelineWithoutDot,
                            tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    llvm::MapVector<Operation *, int> loadOpToIndLevel;
    DenseSet<Operation *> seen;
    DenseSet<Operation *> excluded;

    std::function<void(Operation *, Operation *, int)> dfs =
        [&](Operation *op, Operation *finalUser, int distance) {
          if (!seen.insert(op).second || excluded.count(op))
            return;
          if (isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(
                  op)) {
            if (!isPipeliningBeneficial(op, finalUser, axisInfoAnalysis))
              return;
            if (loadOpToIndLevel.count(op)) {
              int level = loadOpToIndLevel[op];
              if (level != distance) {
                // If we have multiple uses at different distances, we don't
                // know which one to pick.
                LDBG("Load " << *op
                             << " has multiple uses at different distances:"
                             << level << " and " << distance);
                loadOpToIndLevel.erase(op);
                excluded.insert(op);
                return;
              }
            } else {
              LDBG("Load " << *op << " considered for pipelining with distance "
                           << distance);
              loadOpToIndLevel[op] = distance;
            }
            finalUser = op;
            distance++;
          }
          for (Value operand : getNestedOperands(op)) {
            if (isa<mlir::triton::DotOpInterface>(op)) {
              // Heuristic: only pipeline A and B operands of the dot op.
              if (operand == op->getOperand(2))
                continue;
            }
            Value v = operand;
            Operation *defOp = v.getDefiningOp();
            if (defOp && defOp->getBlock() == op->getBlock()) {
              dfs(defOp, finalUser, distance);
            }
          }
        };

    bool seenDot = false;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      // Arbitrary heuristic. TMEMStoreOp is included to keep logic consistent
      // with legacy code when we weren't hoisting tmem allocas.
      if (!isa<mlir::triton::DotOpInterface, ttng::TMEMStoreOp>(op))
        continue;
      seenDot = true;
      seen.clear();
      dfs(&op, &op, 0);
    }

    // If the loop has numStages attribute, also consider pipelining other loads
    // that are not directly used by dot ops.
    if (pipelineWithoutDot && !seenDot) {
      for (Operation &op : forOp.getBody()->without_terminator()) {
        if (!isa<tt::LoadOp, tt::DescriptorLoadOp, tt::DescriptorGatherOp>(op))
          dfs(&op, &op, 0);
      }
    }

    return loadOpToIndLevel;
  }
};

class AssignMMALatencies {
public:
  AssignMMALatencies(scf::ForOp forOp, DenseMap<Operation *, int> &opLatency)
      : forOp(forOp), opLatency(opLatency) {};

  void run() {
    DenseMap<Operation *, int> mmaSelfLatency;
    // Check if the load op (mma operand) is pipelineable.
    auto isLoadToBePipelined = [&](Operation *op) {
      return opLatency.count(op) && opLatency[op] > 0;
    };
    for (auto &op : forOp.getBody()->without_terminator()) {
      // If the acc can not be multibuffered, do not pipeline the uses of
      // the MMA to later stages.
      if (auto mma = dyn_cast<ttng::MMAv5OpInterface>(&op)) {
        // Try to push out the wait by one stage even if the operands are not
        // pipelineable, but we know where the loads are scheduled, so we can
        // place the wait right before the loads.

        if (hasSyncDots(forOp)) {
          // Skip pipelining MMA in the loops where sync dots are used. This
          // is a dirty heuristic for performance drops in kernels where we
          // would rather want to have last iteration peeled instead of having a
          // full iteration of masked operations only to execute single wait.
          continue;
        }
        auto pipeHelper = ttng::MMAv5PipelineableOperandsHelper(
            mma, forOp, isLoadToBePipelined);
        if (pipeHelper.isPipelineable ||
            (pipeHelper.isOperandsStateDetermined &&
             !ttng::hasLoadsAfterMMA(mma, forOp))) {
          // MMA can be overlapped with itself
          mmaSelfLatency[mma] = 1;
          if (!ttng::requiresAccMultiBuffering(mma, forOp) ||
              (ttng::isAccMultibufferingPossible(mma, forOp) &&
               !getDisallowAccMultiBuffer(forOp))) {
            // MMA's users can be pushed to the next stage
            opLatency[&op] = 1;
          }
          // HACK: A pipelined MMA's latency should equal the number of buffers
          // for the accumulator, but when the user is in an `scf.if` in SWP,
          // the `scf.if` is pushed to the end of the loop rather than peeled
          // before the MMA op, requiring an extra buffer due to liverange
          // overlap. WS does not have this problem because the MMA is placed in
          // a different partition than the MMA, so we can correctly set the
          // latency.
          if (forOp->hasAttr(kWarpSpecializeAttrName)) {
            if (ttng::hasAccReadModifyWrite(mma, forOp))
              opLatency.erase(&op); // can't pipeline the MMA
            else
              opLatency[&op] += 1;
          }
        }
      }
    }
    serializeSelfLatencies(forOp->getParentOfType<ModuleOp>(), mmaSelfLatency);
  }

private:
  scf::ForOp forOp;
  DenseMap<Operation *, int> &opLatency;

  bool hasSyncDots(scf::ForOp forOp) {
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (isa<mlir::triton::DotOp>(op))
        return true;
    }
    return false;
  }
};

// Discover operations that should become async and assign latencies to them
// based on the numStages value provided by the user.
//
// Look for load ops that directly or indirectly feed into dot ops. Based on the
// requested number of stages assign the latencies in a way that cover all the
// stages with the sum of latencies in the chain from the first load to the
// final dot op.
void assignLatencies(ModuleOp moduleOp, int defaultNumStages) {
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) {
    // Bail out for loops with num_stage <= 1.
    if (preCondition(forOp) &&
        getNumStagesOrDefault(forOp, defaultNumStages) > 1)
      loops.push_back(forOp);
  });
  if (loops.empty())
    return;

  DenseMap<Operation *, int> opLatency;
  for (auto forOp : loops) {
    if (hasLatenciesAssigned(forOp)) {
      assignUserProvidedLatencies(forOp, opLatency);
      continue;
    }
    int numStages = getNumStagesOrDefault(forOp, defaultNumStages);
    AssignLoadLatencies(forOp, numStages, opLatency).run();
    AssignMMALatencies(forOp, opLatency).run();
  }
  serializeLatencies(moduleOp, opLatency);
}

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_TRITONGPUASSIGNLATENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

struct AssignLatencies
    : public impl::TritonGPUAssignLatenciesBase<AssignLatencies> {
  using TritonGPUAssignLatenciesBase::TritonGPUAssignLatenciesBase;

  void runOnOperation() override { assignLatencies(getOperation(), numStages); }
};

} // namespace mlir::triton::gpu
