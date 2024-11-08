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

#define GEN_PASS_DEF_TRITONGPUPIPELINESCHEDULER
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {
// Return true if the preconditions for pipelining the loop are met.
bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return false;
  // Don't pipeline outer loops.
  if (forOp
          ->walk([&](Operation *op) {
            if (forOp.getOperation() == op)
              return WalkResult::advance();
            if (isa<scf::ForOp, scf::WhileOp>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  return true;
}

// TODO pawel: unify with MatmulLoopPipeline.cpp
// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return the shared encoding that needs to be
// used to be compatible with users' layouts. If there are imcompatible shared
// encodings, raise assertion, since incompatible shared encoding has been
// handled in splitLoadsForIncompatible.
static std::optional<ttg::SharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value val) {
  ttg::SharedEncodingAttr attr;
  for (Operation *user : val.getUsers()) {
    ttg::SharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return std::nullopt;
    if (auto memDesc =
            dyn_cast<triton::MemDescType>(user->getResult(0).getType())) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(user->getResult(0)).has_value())
        return std::nullopt;
    } else {
      if (!isa<ttg::LocalLoadOp, ttg::ConvertLayoutOp>(user))
        return std::nullopt;
      auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
          cast<TensorOrMemDesc>(user->getResult(0).getType()).getEncoding());
      if (!dotOpEnc)
        return std::nullopt;
      auto srcTy = cast<TensorOrMemDesc>(val.getType());
      auto CTALayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = ttg::getOrder(srcTy.getEncoding());
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      tempAttr = ttg::SharedEncodingAttr::get(
          val.getContext(), dotOpEnc, srcTy.getShape(), order, CTALayout,
          bitWidth, /*needTrans=*/false);
    }
    // Check that the shared encodings needed by the users are compatible.
    if (attr != nullptr && attr != tempAttr) {
      return std::nullopt;
    }
    attr = tempAttr;
  }
  return attr;
}

bool canHaveSharedEncoding(Operation *op, Operation *user) {
  // TMA loads load into shared memory, has shared encoding
  // by definition.
  if (isa<tt::ExperimentalDescriptorLoadOp>(op))
    return true;
  auto loadOp = cast<tt::LoadOp>(op);
  auto dst = loadOp.getResult();
  // Load used for initializing shared memory.
  // TODO: We could check if there are multiple users and
  // if they all have the same encoding.
  if (dst.hasOneUse() && isa<ttg::LocalAllocOp>(*dst.getUsers().begin()))
    return true;
  if (getSharedEncIfAllUsersAreDotEnc(dst).has_value())
    return true;
  // TODO: Original pipeliner also checks if generic shared encoding can be
  // applied. Figure out how much of this is necessary.
  return false;
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

bool loadGoodForPipelining(Operation *op, Operation *user,
                           tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
    if (isSmallLoad(loadOp, axisInfoAnalysis))
      return false;
  }
  if (!canHaveSharedEncoding(op, user)) {
    return false;
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

  std::function<void(Operation * op, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *user) {
        if (!seen.insert(op).second)
          return;
        if (isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op)) {
          if (!loadGoodForPipelining(op, user, axisInfoAnalysis))
            return;
          if (loadOpToIndLevel.count(op)) {
            int level = loadOpToIndLevel[op];
            if (level != distance) {
              // If we have multiple uses at different distances, we don't know
              // which one to pick.
              loadOpToIndLevel.erase(op);
            }
          } else {
            loadOpToIndLevel[op] = distance;
          }
          distance++;
        }
        for (Value operand : op->getOperands()) {
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, distance, user);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!op.hasTrait<OpTrait::DotLike>())
      continue;
    seen.clear();
    dfs(&op, 0, nullptr);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (pipelineWithoutDot) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp, tt::ExperimentalDescriptorLoadOp>(op))
        dfs(&op, 0, nullptr);
    }
  }

  return loadOpToIndLevel;
}

} // namespace

// Look for load ops that directly or indirectly feed into dot ops. Based
// on the requested number of stages assign the latencies in a way that
// cover all the stages with the sum of latencies in the chain from the first
// load to the final dot op.
DenseMap<Operation *, int> assignLatencies(scf::ForOp forOp, int numStages) {
  bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
  ModuleOp moduleOp = forOp->getParentOfType<ModuleOp>();
  tt::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);

  llvm::MapVector<Operation *, int> loadOpToIndLevel =
      loadOpsToIndirectionLevel(forOp, pipelineWithoutDot, axisInfoAnalysis);
  if (loadOpToIndLevel.empty())
    return DenseMap<Operation *, int>();

  // We assume loads with different dist are assigned to different stages.
  // If numStages is 2, we will have no stage available for indirect loads
  // with dist >= 1. In general, when dist is equal to numStages - 1, we
  // should not pipeline it.
  auto it = llvm::remove_if(loadOpToIndLevel, [=](auto op) {
    return std::get<1>(op) >= numStages - 1;
  });

  // Calculate the stage distance between applicable loads.
  auto vals = llvm::make_second_range(loadOpToIndLevel);
  int maxIndirectionLevel =
      vals.empty() ? 0 : *std::max_element(vals.begin(), vals.end());
  unsigned loadLatency = (numStages - 1) / (maxIndirectionLevel + 1);

  DenseMap<Operation *, int> opLatency;
  for (auto [loadOp, dist] : loadOpToIndLevel) {
    opLatency[loadOp] = loadLatency;
  }
  return opLatency;
}

void scheduleLoop(scf::ForOp forOp, int numStages) {
  if (!preCondition(forOp))
    return;

  // 1. Assign latencies
  //    Go over the interesting ops and assign latencies (based on the
  //    numStages) to the them, trying to populate the allowed stages. This step
  //    will be at some point extracted to separate pass that will be run only
  //    for loops missing the latency information.
  DenseMap<Operation *, int> opLatency = assignLatencies(forOp, numStages);

  // numStages should not be used below this point. We should know everything
  // based on the assigned stages

  // 2. Schedule key ops
  //    Based on the latencies, schedule the key ops to the stages.
  // 3. Schedule dependencies
  //    Schedule the dependencies (regular and dist 1)
  // 4. Schedule the rest of the ops to the last stage
}

struct PipelineScheduler
    : public impl::TritonGPUPipelineSchedulerBase<PipelineScheduler> {
  using impl::TritonGPUPipelineSchedulerBase<
      PipelineScheduler>::TritonGPUPipelineSchedulerBase;

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
      scheduleLoop(forOp, loopNumStages);
    }
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
