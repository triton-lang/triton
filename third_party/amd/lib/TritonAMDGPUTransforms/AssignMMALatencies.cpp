#include "TritonAMDGPUTransforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonamdgpu-assign-mma-latencies"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace {

//===----------------------------------------------------------------------===//
// assignLatencies
//===----------------------------------------------------------------------===//

// Return true if the preconditions for pipelining the loop are met.
bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO: relax the constraint in the expander.
  if (triton::loopHasDistGreaterThanOne(forOp))
    return false;
  // Don't pipeline outer loops.
  if (triton::isOuterLoop(forOp))
    return false;
  return true;
}

bool hasLatenciesAssigned(scf::ForOp forOp) {
  auto helper = triton::TritonDialect::getLoaded(forOp)->getLatencyAttrHelper();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (helper.getAttr(&op))
      return true;
  }
  return false;
}

void assignUserProvidedLatencies(scf::ForOp forOp,
                                 DenseMap<Operation *, int> &opLatency) {
  auto helper = triton::TritonDialect::getLoaded(forOp)->getLatencyAttrHelper();
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto latencyAttr = helper.getAttr(&op)) {
      opLatency[&op] = latencyAttr.getInt();
    }
  }
}

class AssignMMALatencies {
public:
  AssignMMALatencies(scf::ForOp forOp, DenseMap<Operation *, int> &opLatency)
      : forOp(forOp), opLatency(opLatency) {};

  void run() {
    // Check if the load op (mma operand) is pipelineable.
    auto isLoadToBePipelined = [&](Operation *op) {
      return opLatency.count(op) && opLatency[op] > 0;
    };
    for (auto &op : forOp.getBody()->without_terminator()) {
      // If the acc can not be multibuffered, do not pipeline the uses of
      // the MMA to later stages.
      if (auto mma = dyn_cast<tt::DotOp>(&op)) {
        opLatency[&op] += 1;
      }
    }
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

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_TRITONAMDGPUASSIGNMMALATENCIES
#include "TritonAMDGPUTransforms/Passes.h.inc"

struct AssignLatencies
    : public impl::TritonAMDGPUAssignMMALatenciesBase<AssignLatencies> {
  using TritonAMDGPUAssignMMALatenciesBase::TritonAMDGPUAssignMMALatenciesBase;

  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    auto moduleOp = getOperation();
    moduleOp->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (preCondition(forOp) &&
          triton::getNumStagesOrDefault(forOp, numStages) > 1)
        loops.push_back(forOp);
    });
    if (loops.empty())
      return;

    DenseMap<Operation *, int> opLatency;
    for (auto forOp : loops) {
      AssignMMALatencies(forOp, opLatency).run();
    }
    triton::serializeLatencies(moduleOp, opLatency);
  }
};

} // namespace mlir
