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
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#define DEBUG_TYPE "triton-loop-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace mlir {
namespace triton {
namespace gpu {

// Create a map from load ops to their indirection level and the
// final use of the load op (another load op, or a dot op).
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
static llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
loadOpsToIndirectionLevelAndUse(scf::ForOp forOp) {
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse;
  DenseSet<Operation *> seen;

  std::function<void(Operation * op, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        if (!seen.insert(op).second)
          return;
        if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op)) {
          // TODO: What if there are multiple uses at different distances?
          loadOpToIndLevelAndUse.push_back(std::make_tuple(op, distance, use));
          use = op;
          distance++;
        }
        for (Value operand : op->getOperands()) {
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, distance, use);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!op.hasTrait<OpTrait::DotLike>())
      continue;
    seen.clear();
    dfs(&op, 0, &op);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (forOp->hasAttr(tt::kNumStagesAttrName)) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op))
        dfs(&op, 0, &op);
    }
  }

  return loadOpToIndLevelAndUse;
}

static bool hasSharedEncodingHelper(Operation *loadOp) {
  // If the load is used by a LocalAllocOp, use the same encoding as the allocs.
  // If the allocs don't all have the same encoding, bail.
  if (llvm::any_of(loadOp->getUsers(), [&](Operation *user) {
        return isa<ttg::LocalAllocOp>(user);
      })) {
    ttg::SharedEncodingAttr localAllocEnc;
    for (auto user : loadOp->getUsers()) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
      if (!localAlloc)
        continue;
      auto enc = mlir::cast<ttg::SharedEncodingAttr>(
          localAlloc.getType().getEncoding());
      if (!localAllocEnc) {
        localAllocEnc = enc;
      }
      if (enc != localAllocEnc)
        return false;
    }
    return true;
  }
  return true;
}

// Check to see if loads can be pipelined.
static llvm::DenseSet<Operation *>
filterPipelinedLoad(llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
                        &loadOpToIndLevelAndUse,
                    tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  llvm::DenseSet<Operation *> loadsToPipeline;
  for (auto &[op, dist_, use] : loadOpToIndLevelAndUse) {
    if (loadsToPipeline.count(op))
      // TODO pawel: err, we'd need to verify that the distance is the same
      continue;

    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      assert(!isLoadFromTensorPtr(loadOp) &&
             "Block ptr should have been lowered before this pass.");
      auto ptr = loadOp.getPtr();
      unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

      auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorTy)
        continue;
      auto ty =
          cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
      unsigned width = vec * ty.getIntOrFloatBitWidth();

      // We do not pipeline all loads for the following reasons:
      // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8, or 16.
      // 2. It's likely that pipling small loads won't offer much performance
      //    improvement and may even hurt performance by increasing register
      //    pressure.
      LDBG("Load " << *loadOp << " has width " << width);
      if (width < 32)
        continue;
    }

    bool hasSharedEncoding = false;
    if (use->hasTrait<OpTrait::DotLike>()) {
      if (loadIsMMAv3(op)) {
        hasSharedEncoding = true;
      } else if (isa<tt::ExperimentalDescriptorLoadOp>(op)) {
        hasSharedEncoding = true;
      } else if (auto dot = dyn_cast<tt::DotOp>(use)) {
        // FIXME: if we have a better solution in handling incompatible shared
        // encoding, we can simplify the logic here by checking if all users are
        // dot encoding. Fow now, getSharedEncIfAllUsersAreDotEnc will be used
        // during both scheduling and lowering.
        bool incompatible = false;
        auto sharedEncoding =
            getSharedEncIfAllUsersAreDotEnc(op->getResult(0), incompatible)
                .value_or(nullptr);
        hasSharedEncoding = sharedEncoding != nullptr;
        // If we can't agree on a shared encoding skip pipelinig the load.
        if (incompatible)
          continue;
      }
    } else if (auto loadOp = dyn_cast<tt::LoadOp>(use)) {
      // The use of this loadOp is another loadOp. If the use is not in the
      // loadsToPipeline already, it means that the use is not valid for
      // pipelining for some reason. We should skip this loadOp, too. Note that
      // we have an assumption that distAndUse.second (i.e. the use of this
      // loadOp) has already be processed in a previous loop iteration. This
      // assumption is held by how loadOpsToIndirectionLevelAndUse recursively
      // collects loadOpToIndLevelAndUse using DFS.
      if (loadsToPipeline.count(loadOp) == 0) {
        continue;
      }
    }

    // If we still don't have a shared encoding, try a "generic" shared
    // encoding.
    if (!hasSharedEncoding && !isa<ttng::WarpGroupDotOp>(use))
      hasSharedEncoding = hasSharedEncodingHelper(op);

    // If that still didn't work, bail on pipelining this load.
    if (!hasSharedEncoding) {
      continue;
    }
    loadsToPipeline.insert(op);
  }
  return loadsToPipeline;
}

static void scheduleLoads(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                          DenseSet<Operation *> &rootUsers, int numStages) {

  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse = loadOpsToIndirectionLevelAndUse(forOp);
  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
    for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
      LDBG("  - load: " << *l);
      LDBG("    at indirection level: " << i);
      LDBG("    used by op: " << *u);
    }
  });
  if (loadOpToIndLevelAndUse.empty())
    return;

  // We assume loads with different dist are assigned to different stages.
  // If numStages is 2, we will have no stage available for indirect loads
  // with dist >= 1. In general, when dist is equal to numStages - 1, we
  // should not pipeline it.
  auto it = llvm::remove_if(loadOpToIndLevelAndUse, [=](auto op) {
    return std::get<1>(op) >= numStages - 1;
  });
  loadOpToIndLevelAndUse.erase(it, loadOpToIndLevelAndUse.end());

  // Check which loads are good for pipelining.
  llvm::DenseSet<Operation *> loadsToPipeline =
      filterPipelinedLoad(loadOpToIndLevelAndUse, axisInfoAnalysis);
  if (loadsToPipeline.empty())
    return;

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
    if (loadsToPipeline.count(loadOp) == 0)
      continue;
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);
  }
  unsigned stagesBetweenLoads =
      ceil<unsigned>(numStages - 2, maxIndirectionLevel + 1);

  tt::CoarseSchedule::Cluster rootUsersCluster = schedule.clusters.newAtFront();
  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    if (loadsToPipeline.count(loadOp) == 0)
      continue;
    // Non-LoadOp(s) are the root uses of all LoadOp(s) and should be
    // always present in the opInfo
    if (!isa<tt::LoadOp>(use)) {
      schedule.insert(use, numStages - 1, rootUsersCluster);
      rootUsers.insert(use);
    }
  }

  SmallVector<tt::CoarseSchedule::Cluster> loadsClusters;
  for (int i = 0; i < maxIndirectionLevel + 1; i++) {
    loadsClusters.push_back(schedule.clusters.newAtBack());
  }
  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    if (loadsToPipeline.count(loadOp) == 0)
      continue;
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    schedule.insert(loadOp, stage, loadsClusters[indLevel]);
  }
}

// Schedule the prologue and epilogue `if` ops in the loop, pushing them as
// close to the loop boundaries as possible. Return the cluster after the
// prologue (or the beginning of the loop if there is no prologue).
static tt::CoarseSchedule::Cluster
schedulePrologueAndEpilogue(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                            DenseSet<Operation *> &rootUsers, int numStages) {
  tt::CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : schedule.getOpsInOrder(forOp)) {
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice((Operation *)op, &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  tt::CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
  for (auto [ifOp, stage] : ifsToStage) {
    schedule.insert(ifOp, stage, prologueCluster);
  }

  // Look for the IfOp that is in the forward slice of the root users and put it
  // at the end of the loop.
  tt::CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto rootUser : rootUsers) {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(rootUser, &forwardSlice);

    int stage = schedule[rootUser].first;
    for (auto op : forwardSlice) {
      scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp == nullptr) {
        // check if the op is in the body of an if op that's part of the loop
        auto parentOp = op->getParentOp();
        if (parentOp != nullptr &&
            parentOp->getParentOp() == forOp.getOperation()) {
          ifOp = dyn_cast<scf::IfOp>(parentOp);
        }
      }
      if (ifOp) {
        schedule.insertIfAbsent(ifOp, stage,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
static void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                            tt::CoarseSchedule &schedule,
                                            int numStages) {
  auto getNestedOperands = [](Operation *op) -> SmallVector<Value> {
    SmallVector<Value> operands;
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (operand.getParentBlock()->getParentOp()->isAncestor(nestedOp))
          operands.push_back(operand);
      }
    });
    return operands;
  };

  // Mapping from the cluster to the cluster before it.
  DenseMap<tt::CoarseSchedule::Cluster *, tt::CoarseSchedule::Cluster>
      dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && schedule.count(defOp) == 0) {
            if (isa<tt::LoadOp>(defOp)) {
              // Exception: Schedule loads with a distance of 1 together
              // with the current op.
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster, true);
            } else {
              if (dist1Cluster.count(&cluster) == 0) {
                dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
              }
              schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
              schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster],
                                      true);
            }
          }
        }
      }
    }
  }
}

static void
scheduleRemainingToLastStage(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                             tt::CoarseSchedule::Cluster afterPrologue,
                             int numStages) {
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  DenseMap<Operation *, tt::CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, cluster] : schedule.getOpsInOrder(forOp)) {
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        tt::CoarseSchedule::Cluster userCluster = opToCluster[user];
        tt::CoarseSchedule::Cluster opCluster;
        if (schedule.count(op))
          opCluster = schedule[op].second;
        else
          opCluster = opToCluster[op];
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
}

static const char *kLoopScheduleAttrName = "tt.loop_schedule";
std::string getLoopScheduleOrDefault(scf::ForOp forOp) {
  if (!forOp->hasAttr(kLoopScheduleAttrName))
    return "default";
  return (cast<StringAttr>(forOp->getAttr(kLoopScheduleAttrName))).str();
}

static bool isHeavyComputation(Operation *op) {
  // include exp2, mulf, addf 1D. Somehow we don't go through reduction
  // when checking dependencies
  if (!isa<arith::MulFOp>(op) && !isa<math::Exp2Op>(op) &&
      !isa<arith::AddFOp>(op))
    return false;
  auto tensorTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!tensorTy)
    return false;
  if (tensorTy.getRank() < 1)
    return false;
  return true;
}

#define GEN_PASS_DEF_TRITONGPULOOPSCHEDULING
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPULoopSchedulingPass
    : public impl::TritonGPULoopSchedulingBase<TritonGPULoopSchedulingPass> {
public:
  using impl::TritonGPULoopSchedulingBase<
      TritonGPULoopSchedulingPass>::TritonGPULoopSchedulingBase;

  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return numStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  }

  bool
  isFlashAttention(scf::ForOp forOp,
                   llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
                       &loadOpToIndLevelAndUse,
                   SmallVector<Operation *> &keyOps,
                   DenseSet<Operation *> &heavyCompOps) {
    SmallVector<Operation *> loads;
    SmallVector<Operation *> dots;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      // Check for loop-carried dependencies.
      // We have two loadOps, one feeding the first dot, and the other feeding
      // the second dot.
      if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op)) {
        loads.push_back(&op);
      }
      if (op.hasTrait<OpTrait::DotLike>()) {
        dots.push_back(&op);
      }
    }
    if (dots.size() != 2 || loads.size() != 2)
      return false;

    Operation *secondDot = dots[1];
    DenseSet<Operation *> seen;
    DenseSet<Operation *> tracedDots;
    // Make sure there is a dependency path from firstDot to secondDot.
    // This means we need to do computation pipelining to break the dependency.
    std::function<void(Operation * op)> dfs = [&](Operation *op) {
      if (!seen.insert(op).second)
        return;
      for (Value operand : op->getOperands()) {
        Value v = operand;
        Operation *defOp = v.getDefiningOp();
        if (defOp && defOp->getBlock() == op->getBlock()) {
          if (defOp->hasTrait<OpTrait::DotLike>()) {
            // Stop tracing when hitting a dot.
            tracedDots.insert(defOp);
          } else {
            if (isHeavyComputation(defOp))
              heavyCompOps.insert(defOp);
            dfs(defOp);
          }
        }
      }
    };
    dfs(secondDot);
    if (tracedDots.size() != 1)
      return false;

    for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
      if (dist != 0)
        return false;
    }

    keyOps.push_back(loads[0]); // FIXME
    keyOps.push_back(loads[1]);
    keyOps.push_back(dots[0]);
    keyOps.push_back(secondDot);
    return true;
  }

  void getFAFirstDotSchedule(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                             int numStages) {
    llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
        loadOpToIndLevelAndUse = loadOpsToIndirectionLevelAndUse(forOp);
    LLVM_DEBUG({
      LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
      for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
        LDBG("  - load: " << *l);
        LDBG("    at indirection level: " << i);
        LDBG("    used by op: " << *u);
      }
    });
    if (loadOpToIndLevelAndUse.empty())
      return;

    // Check to see if the for loop matches the pattern for flash attention.
    // If yes, move the first dot to its own stage (numStages - 2), the
    // rest of the computation will be in stage (numStages - 1). The two loads
    // will be in stage 0 and 1.
    SmallVector<Operation *> keyOps;
    DenseSet<Operation *> heavyCompOps;
    if (!isFlashAttention(forOp, loadOpToIndLevelAndUse, keyOps,
                          heavyCompOps)) {
      LDBG("isFlashAttention returns false");
      return;
    }
    // firstLoad: keyOps[0]
    tt::CoarseSchedule::Cluster rootUsersCluster =
        schedule.clusters.newAtFront();
    tt::CoarseSchedule::Cluster loadCluster = schedule.clusters.newAtBack();
    schedule.insert(keyOps[0], 0, loadCluster);
    schedule.insert(keyOps[1], 1, loadCluster);
    schedule.insert(keyOps[2], numStages - 2, rootUsersCluster);
    schedule.insert(keyOps[3], numStages - 1, rootUsersCluster);
    return;
  }

  void getFASecondDotSchedule(scf::ForOp forOp, tt::CoarseSchedule &schedule,
                              int numStages) {
    llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
        loadOpToIndLevelAndUse = loadOpsToIndirectionLevelAndUse(forOp);
    LLVM_DEBUG({
      LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
      for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
        LDBG("  - load: " << *l);
        LDBG("    at indirection level: " << i);
        LDBG("    used by op: " << *u);
      }
    });
    if (loadOpToIndLevelAndUse.empty())
      return;

    // Check to see if the for loop matches the pattern for flash attention.
    // If yes, move the second dot to its own stage (numStages - 1), the
    // rest of the computation will be in stage (numStages - 2). The two loads
    // will be in stage 0 and 1.
    SmallVector<Operation *> keyOps;
    DenseSet<Operation *> heavyCompOps;
    if (!isFlashAttention(forOp, loadOpToIndLevelAndUse, keyOps,
                          heavyCompOps)) {
      LDBG("isFlashAttention returns false");
      return;
    }
    // Go through loop body
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (isHeavyComputation(&op))
        heavyCompOps.insert(&op);
    }
    // keyOps: load0, load1, dot0, dot1
    //   Dot0(i+1)
    //   Dot1(i)
    //   Softmax(i+1): includes MUL0(i+1)
    //   MUL1(i+1)
    tt::CoarseSchedule::Cluster rootUsersCluster =
        schedule.clusters.newAtFront();
    tt::CoarseSchedule::Cluster nextCluster = schedule.clusters.newAtBack();
    tt::CoarseSchedule::Cluster nextNextCluster = schedule.clusters.newAtBack();
    tt::CoarseSchedule::Cluster loadCluster = schedule.clusters.newAtBack();
    schedule.insert(keyOps[0], 0, loadCluster);
    schedule.insert(keyOps[1], 1, loadCluster);
    schedule.insert(keyOps[2], numStages - 2, rootUsersCluster);
    schedule.insert(keyOps[3], numStages - 1, nextCluster);
    // Softmax(i+1), MUL1(i+1) in nextNextCluster
    for (auto *heavyOp : heavyCompOps)
      schedule.insert(heavyOp, numStages - 2, nextNextCluster);
    return;
  }

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (getNumStagesOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    if (loops.empty())
      return;
    for (scf::ForOp forOp : loops) {
      int loopNumStages = getNumStagesOrDefault(forOp);
      DenseSet<Operation *> rootUsers;
      tt::CoarseSchedule coarseSchedule(loopNumStages);
      std::string loopSchedule = getLoopScheduleOrDefault(forOp);
      if (loopSchedule == "default") {
        scheduleLoads(forOp, coarseSchedule, rootUsers, loopNumStages);
      } else if (loopSchedule == "FA_firstDot") {
        getFAFirstDotSchedule(forOp, coarseSchedule, loopNumStages);
      } else if (loopSchedule == "FA_secondDot") {
        getFASecondDotSchedule(forOp, coarseSchedule, loopNumStages);
      } else {
        assert(false && "unrecognized loop schedule");
      }
      if (coarseSchedule.opToStageAndCluster.size() == 0)
        continue;
      tt::CoarseSchedule::Cluster afterPrologue = schedulePrologueAndEpilogue(
          forOp, coarseSchedule, rootUsers, loopNumStages);

      scheduleDependencies(forOp, coarseSchedule, loopNumStages);
      LLVM_DEBUG({
        LDBG("Coarse schedule with dependencies:");
        coarseSchedule.dump();
      });

      scheduleDistanceOneDependencies(forOp, coarseSchedule, loopNumStages);
      LLVM_DEBUG({
        LDBG("Coarse schedule with dist 1:");
        coarseSchedule.dump();
      });

      LDBG("afterPrologue = " << *afterPrologue);
      scheduleRemainingToLastStage(forOp, coarseSchedule, afterPrologue,
                                   loopNumStages);
      LLVM_DEBUG({
        LDBG("Final coarse schedule:");
        coarseSchedule.dump();
      });

      // Go through schedule and assign (stage, cluster).
      // shift so afterPrologue will be at clusterId 0
      auto ctx = forOp.getContext();
      for (auto [op, stage_, cluster] : coarseSchedule.getOpsInOrder(forOp)) {
        op->setAttr(mlir::triton::kLoopStageAttrName,
                    IntegerAttr::get(IntegerType::get(ctx, 32), stage_));
        op->setAttr(mlir::triton::kLoopClusterAttrName,
                    IntegerAttr::get(IntegerType::get(ctx, 32),
                                     *cluster /*- *afterPrologue*/));
        LLVM_DEBUG({
          LDBG("set stage " << stage_ << " cluster " << (*cluster));
          op->dump();
        });
      }
    }
    return;
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
