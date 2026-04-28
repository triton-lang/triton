#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "tritonamdgpu-warp-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUWARPPIPELINE
#include "TritonAMDGPUTransforms/Passes.h.inc"

// Create a scf.execute_region op representing a pipeline cluster.
static void createClusterOp(OpBuilder &b, Location loc,
                            SmallVector<Operation *> &ops,
                            std::pair<StringAttr, int> marker) {
  assert(!ops.empty() && "empty stage");

  // Insert the execute_region before the first op in the cluster.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPoint(ops.front());

  // Build fast ops lookup for the cluster.
  llvm::SmallPtrSet<Operation *, 32> opsLookup(ops.begin(), ops.end());

  // Determine which results have users outside the cluster.
  SmallVector<Type> yieldedTypes;
  SmallVector<std::pair<OpResult, unsigned>>
      resultToYieldIdx; // (orig result, idx in yields)

  for (Operation *op : ops) {
    for (OpResult r : op->getResults()) {
      bool hasExternalUse = llvm::any_of(r.getUsers(), [&](Operation *u) {
        return !opsLookup.count(u) && u->getParentOp() != nullptr;
      });
      if (hasExternalUse) {
        yieldedTypes.push_back(r.getType());
        resultToYieldIdx.emplace_back(
            r, static_cast<unsigned>(yieldedTypes.size() - 1));
      }
    }
  }

  // Create the execute_region with the final result types.
  auto exec = scf::ExecuteRegionOp::create(b, loc, yieldedTypes);
  Block *body = &exec.getRegion().emplaceBlock();
  b.setInsertionPointToStart(body);

  // Clone ops in order, remapping intra-cluster defs to their clones.
  IRMapping mapping;
  for (Operation *op : ops) {
    Operation *clone = b.clone(*op, mapping);
    // Map each result so subsequent clones use the cloned defs.
    for (auto [origRes, clonedRes] :
         llvm::zip(op->getResults(), clone->getResults()))
      mapping.map(origRes, clonedRes);
  }

  // Build the yield values.
  SmallVector<Value> yieldVals(yieldedTypes.size());
  for (auto [origRes, yieldIdx] : resultToYieldIdx) {
    Value mapped = mapping.lookupOrNull(origRes);
    assert(mapped && "mapped result missing");
    yieldVals[yieldIdx] = mapped;
  }
  scf::YieldOp::create(b, loc, yieldVals);

  // Replace external uses of original results with exec results.
  // Internal uses were already remapped when cloning.
  for (auto [origRes, yieldIdx] : resultToYieldIdx) {
    Value repl = exec.getResult(yieldIdx);
    origRes.replaceUsesWithIf(repl, [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return owner && owner->getParentOp() != exec;
    });
  }

  // Erase original ops now that their external uses are redirected.
  std::reverse(ops.begin(), ops.end());
  for (Operation *op : ops)
    op->erase();

  // Keep the region structured for later conversion.
  exec.setNoInline(true);
  exec->setAttr("triton.warp_pipeline.stage", marker.first);
  if (marker.second > -1) {
    exec->setAttr("triton.warp_pipeline.priority",
                  b.getI32IntegerAttr(marker.second));
  }

  LLVM_DEBUG(llvm::dbgs() << "[warp-pipeline] created stage with " << ops.size()
                          << " ops and " << yieldedTypes.size() << " yields\n");
  return;
}

// Ops that may appear before or after a stage but not inside one.
// Barrier/wait still require an explicit border op to split clusters.
static bool canSitBetweenStages(Operation *op) {
  return isa<ttg::AsyncWaitOp, gpu::BarrierOp, triton::gpu::BarrierOp,
             tt::amdgpu::AsyncTDMWait>(op);
}

// Sink pure-scalar ops past adjacent ignorable ops so they join the next
// cluster. After loop unrolling, scalar IV-remap ops (arith.addi/muli) land
// between borders and the ignorable that starts the next iteration (FA
// pattern); without this, WarpPipeliner sees scalars as an incomplete
// cluster when it hits the ignorable and bails out. Single forward pass,
// O(N).
//
// `pending` only accumulates pure scalars and is cleared at any other op,
// so it forms a closed SSA DAG: any use of a pending scalar by the trailing
// ignorable run must be a direct operand, so the dependency check is a
// simple operand scan.
static void sinkPureScalarsPastIgnorables(Block &blk) {
  // Accumulates a run of consecutive pure scalars that might be sunk.
  SmallVector<Operation *> pending;
  auto consumesPending = [&](Operation *user) {
    return llvm::any_of(user->getOperands(), [&](Value v) {
      return llvm::is_contained(pending, v.getDefiningOp());
    });
  };
  for (Operation *op = &blk.front(); op;) {
    Operation *next = op->getNextNode();
    if (triton::isPureScalarOp(op)) {
      pending.push_back(op);
      op = next;
      continue;
    }
    // Non-scalar op: try to sink pending past an ignorable run, then reset.
    if (canSitBetweenStages(op) && !pending.empty()) {
      // Extend `anchor` to the last ignorable in the consecutive run.
      Operation *anchor = op;
      while (anchor->getNextNode() &&
             canSitBetweenStages(anchor->getNextNode()))
        anchor = anchor->getNextNode();
      // Abort the sink if any ignorable in [op..anchor] consumes a pending
      // scalar -- moving its producer past it would break SSA.
      bool conflict = false;
      for (Operation *ign = op; !conflict; ign = ign->getNextNode()) {
        conflict = consumesPending(ign);
        if (ign == anchor)
          break;
      }
      if (!conflict) {
        // Skip past the moved scalars on the next iteration.
        next = anchor->getNextNode();
        // Reverse iteration + moveAfter(anchor) preserves source order:
        // each earlier-inserted scalar is pushed right by later inserts.
        for (Operation *s : llvm::reverse(pending))
          s->moveAfter(anchor);
      }
    }
    // Pending is always cleared at a non-scalar op: the run is broken,
    // either by a successful sink or by an op that anchors them in place.
    pending.clear();
    op = next;
  }
}

// Turns a partitioned region into the warp-pipelined clusters
static LogicalResult createPipeline(OpBuilder &b, Location loc,
                                    scf::ForOp forOp) {
  Block &blk = *forOp.getBody();
  SmallVector<Operation *> cluster;
  SmallVector<std::pair<StringAttr, int>> clusterMarkers;
  SmallVector<SmallVector<Operation *>> clusters;
  auto ctx = forOp.getContext();

  auto isBorder = [](Operation *op) {
    return op->hasAttr("triton.warp_pipeline.border");
  };

  sinkPureScalarsPastIgnorables(blk);

  // One pass over the body; collect clusters split by explicit borders.
  for (Operation &opRef : llvm::make_early_inc_range(blk)) {
    Operation *op = &opRef;
    if (isBorder(op)) { // Wrap-up one cluster at a border.
      StringAttr clusterStr =
          op->getAttrOfType<StringAttr>("triton.warp_pipeline.border");
      int priority = -1;
      if (auto intAttr =
              op->getAttrOfType<IntegerAttr>("triton.warp_pipeline.priority")) {
        priority = intAttr.getInt();
      }
      clusterMarkers.push_back({clusterStr, priority});
      if (cluster.empty()) {
        // This allows user to deliberately insert a pipeline bubble with a
        // cluster only contains a dummy operation.
        b.setInsertionPoint(op);
        auto dummyOp = ROCDL::SchedBarrier::create(b, loc, 0);
        dummyOp->setAttr("triton.warp_pipeline.empty_cluster", b.getUnitAttr());
        cluster.push_back(dummyOp);
      }
      clusters.push_back(std::move(cluster));
      cluster.clear();
      op->erase(); // remove the marker
      continue;
    }
    if (canSitBetweenStages(op)) {
      // Ignorable ops may appear before or after a stage, but not inside it.
      // If encountered while building an execute_region, reject warp-pipeline.
      if (!cluster.empty())
        return failure();
      continue;
    }
    if (isa<scf::YieldOp>(op)) // End of the loop
      break;

    // Keep collecting ops for a cluster.
    cluster.push_back(op);
  }
  if (!cluster.empty()) { // create the last cluster if needed.
    clusters.push_back(std::move(cluster));
    auto clusterStr = StringAttr::get(ctx, "last_cluster");
    clusterMarkers.push_back({clusterStr, -1});
  }

  // no pipeline clusters detected if 1 or 0 chunk found
  if (clusters.size() < 2)
    return failure();

  // Materialize each cluster as an execute_region.
  int totalStages = clusters.size();
  for (auto &&[stageOps, marker] : llvm::zip(clusters, clusterMarkers)) {
    if (stageOps.empty())
      continue;
    createClusterOp(b, loc, stageOps, marker);
  }

  // Annotate the loop for the backend.
  b.setInsertionPoint(forOp);
  forOp->setAttr("triton.warp_pipeline.pipelined_for", b.getUnitAttr());

  LDBG("[warp-pipeline] total_stages=" << totalStages << "\n");
  return success();
}

struct TritonAMDGPUWarpPipelinePass
    : impl::TritonAMDGPUWarpPipelineBase<TritonAMDGPUWarpPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m);
    for (auto funcOp : m.getOps<tt::FuncOp>()) {
      funcOp.walk([&](scf::ForOp forOp) {
        Location loc = forOp.getLoc();
        if (createPipeline(builder, loc, forOp).failed())
          LDBG("Failed warp-pipelining");
      });
    }
  }
};

} // namespace mlir
