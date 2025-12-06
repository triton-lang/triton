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
                            SmallVector<Operation *> &ops, StringAttr marker) {
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
  exec->setAttr("triton.warp_pipeline.stage", marker);

  LLVM_DEBUG(llvm::dbgs() << "[warp-pipeline] created stage with " << ops.size()
                          << " ops and " << yieldedTypes.size() << " yields\n");
  return;
}

// Turns a partitioned region into the warp-pipelined clusters
static LogicalResult createPipeline(OpBuilder &b, Location loc,
                                    scf::ForOp forOp) {
  // Collect ops in the loop body
  Block &blk = *forOp.getBody();
  SmallVector<Operation *> cluster;
  SmallVector<StringAttr> clusterMarkers;
  SmallVector<SmallVector<Operation *>> clusters;
  auto ctx = forOp.getContext();

  // ops cannot be located within a cluster
  // barrier/wait still require border op
  auto isIgnorable = [](Operation *op) {
    return isa<ttg::AsyncWaitOp, gpu::BarrierOp, tt::amdgpu::AsyncTDMWait>(op);
  };

  auto isBorder = [](Operation *op) {
    return op->hasAttr("triton.warp_pipeline.border");
  };

  // One pass over the body; collect clusters split by explicit borders.
  for (Operation &opRef : llvm::make_early_inc_range(blk)) {
    Operation *op = &opRef;
    if (isBorder(op)) { // Wrap-up one cluster at a border.
      auto clusterStr =
          op->getAttrOfType<StringAttr>("triton.warp_pipeline.border");
      clusterMarkers.push_back(clusterStr);
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
    if (isIgnorable(op)) {
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
    clusterMarkers.push_back(clusterStr);
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
