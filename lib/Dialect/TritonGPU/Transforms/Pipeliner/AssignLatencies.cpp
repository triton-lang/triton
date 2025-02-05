#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-pipeline-schedule"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace gpu {

namespace {

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

bool canHaveSharedEncoding(tt::LoadOp op) {
  // If used by an user with DotOp encoding, all the uses must be compatible.
  bool incompatible = false;
  getSharedEncIfAllUsersAreDotEnc(op.getResult(), incompatible);
  return !incompatible;
}

bool isSmallLoad(tt::LoadOp loadOp,
                 tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  assert(!isLoadFromTensorPtr(loadOp) &&
         "Block ptr should have been lowered before this pass.");
  auto ptr = loadOp.getPtr();
  unsigned vec = axisInfoAnalysis.getPtrContiguity(ptr);
  if (auto mask = loadOp.getMask())
    vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));

  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy)
    return true;
  auto ty = cast<tt::PointerType>(tensorTy.getElementType()).getPointeeType();
  unsigned width = vec * ty.getIntOrFloatBitWidth();

  // We do not pipeline all loads for the following reasons:
  // 1. On nvidia GPUs, cp.async's cp-size can only be 4, 8, or 16.
  // 2. It's likely that pipling small loads won't offer much performance
  //    improvement and may even hurt performance by increasing register
  //    pressure.
  LDBG("Load " << *loadOp << " has width " << width);
  return width < 32;
}

int getCopyVecBytes(RankedTensorType registerTy,
                    ttg::SharedEncodingTrait sharedEnc) {
  auto regLayout = triton::gpu::toLinearLayout(registerTy.getShape(),
                                               registerTy.getEncoding());
  auto sharedLayout =
      triton::gpu::toLinearLayout(registerTy.getShape(), sharedEnc);
  auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);
  const int vecElems = regToSharedLayout.getNumConsecutiveInOut();
  return vecElems * registerTy.getElementTypeBitWidth() / 8;
}

bool isPipeliningBeneficial(Operation *op, Operation *finalUser,
                            tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    if (isSmallLoad(loadOp, axisInfoAnalysis)) {
      LDBG("Load " << *loadOp << " is too small for pipelining");
      return false;
    }
  }
  if (isa<tt::ExperimentalDescriptorLoadOp, tt::ExperimentalDescriptorGatherOp>(
          op))
    return true;
  if (isa<ttng::WarpGroupDotOp>(finalUser) &&
      getMMALoadType(op) == MMALoadType::DoNotPipeline) {
    LDBG("Load " << *op << " used by WarpGroupDotOp with incompatible layout");
    return false;
  }
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
        // If the load is used by a LocalAllocOp, all the users need to have the
        // same encoding.
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
        if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp,
                tt::ExperimentalDescriptorGatherOp>(op)) {
          if (!isPipeliningBeneficial(op, finalUser, axisInfoAnalysis))
            return;
          if (loadOpToIndLevel.count(op)) {
            int level = loadOpToIndLevel[op];
            if (level != distance) {
              // If we have multiple uses at different distances, we don't know
              // which one to pick.
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
        for (Value operand : op->getOperands()) {
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
        if (auto tmemAlloc = dyn_cast<nvidia_gpu::TMEMAllocOp>(op)) {
          if (!tmemAlloc.getSrc()) {
            for (auto user : tmemAlloc.getResult().getUsers()) {
              if (auto tmemCopy = dyn_cast<nvidia_gpu::TMEMCopyOp>(user)) {
                dfs(tmemCopy.getSrc().getDefiningOp(), finalUser, distance);
                break;
              }
            }
          }
        }
      };

  bool seenDot = false;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isa<mlir::triton::DotOpInterface>(op))
      continue;
    seenDot = true;
    seen.clear();
    dfs(&op, &op, 0);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (pipelineWithoutDot && !seenDot) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp,
               tt::ExperimentalDescriptorGatherOp>(op))
        dfs(&op, &op, 0);
    }
  }

  return loadOpToIndLevel;
}

bool hasLatenciesAssigned(scf::ForOp forOp) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (op.hasAttr("tt_latency"))
      return true;
  }
  return false;
}

void assignUserProvidedLatencies(scf::ForOp forOp,
                                 DenseMap<Operation *, int> &opLatency) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto latencyAttr = op.getAttr("tt_latency")) {
      opLatency[&op] = mlir::cast<IntegerAttr>(latencyAttr).getInt();
    }
  }
}

} // namespace

// Look for load ops that directly or indirectly feed into dot ops. Based
// on the requested number of stages assign the latencies in a way that
// cover all the stages with the sum of latencies in the chain from the first
// load to the final dot op.
DenseMap<Operation *, int> assignLatencies(ModuleOp moduleOp,
                                           int defaultNumStages) {
  auto getNumStagesOrDefault = [defaultNumStages](scf::ForOp forOp) -> int {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr(mlir::triton::kNumStagesAttrName))
      return defaultNumStages;
    return mlir::cast<IntegerAttr>(
               forOp->getAttr(mlir::triton::kNumStagesAttrName))
        .getInt();
  };

  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) {
    // Bail out for loops with num_stage <= 1.
    if (preCondition(forOp) && getNumStagesOrDefault(forOp) > 1)
      loops.push_back(forOp);
  });
  if (loops.empty())
    return DenseMap<Operation *, int>();

  DenseMap<Operation *, int> opLatency;
  for (auto forOp : loops) {
    if (hasLatenciesAssigned(forOp)) {
      assignUserProvidedLatencies(forOp, opLatency);
      continue;
    }
    int numStages = getNumStagesOrDefault(forOp);
    bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
    ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
    tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

    llvm::MapVector<Operation *, int> loadOpToIndLevel =
        loadOpsToIndirectionLevel(forOp, pipelineWithoutDot, axisInfoAnalysis);
    if (loadOpToIndLevel.empty())
      continue;

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
  return opLatency;
}

} // namespace gpu
} // namespace triton
} // namespace mlir
