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
                          int numStages) {

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
      tt::CoarseSchedule coarseSchedule(loopNumStages);
      scheduleLoads(forOp, coarseSchedule, loopNumStages);
      if (coarseSchedule.opToStageAndCluster.size() == 0)
        continue;
      tt::CoarseSchedule::Cluster afterPrologue =
          schedulePrologueAndEpilogue(forOp, coarseSchedule);

      scheduleDependencies(forOp, coarseSchedule);
      LLVM_DEBUG({
        LDBG("Coarse schedule with dependencies:");
        coarseSchedule.dump();
      });

      scheduleDistanceOneDependencies(forOp, coarseSchedule);
      LLVM_DEBUG({
        LDBG("Coarse schedule with dist 1:");
        coarseSchedule.dump();
      });

      LDBG("afterPrologue = " << *afterPrologue);
      scheduleRemainingToLastStage(forOp, coarseSchedule, afterPrologue);
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
