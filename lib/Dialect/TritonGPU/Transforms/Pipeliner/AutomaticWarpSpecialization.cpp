#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUAUTOMATICWARPSPECIALIZATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct AutomaticWarpSpecialization
    : triton::gpu::impl::TritonGPUAutomaticWarpSpecializationBase<
          AutomaticWarpSpecialization> {
  using TritonGPUAutomaticWarpSpecializationBase::
      TritonGPUAutomaticWarpSpecializationBase;

  void runOnOperation() override;
};
} // namespace

void AutomaticWarpSpecialization::runOnOperation() {
  // Collect for loops to warp specialize. This pass expects the loop to already
  // be scheduled.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kLatenciesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
  }

  // Analyze partitions and organized them into a DAG. Each partition is a node
  // with multiple inputs and multiple outputs. Entry partitions have no inputs
  // and exit partitions have no outputs. The DAG is determined based on SSA
  // dependencies. Each output has a latency T that determines how its buffered.
  // I.e. an output with latency T will be buffered for T cycles. A partition
  // can have outputs with different latencies. Ops without latency/partition
  // assginments are assumed to be "free" and can be cloned as necessary.

  // - Ops are assigned to partitions.
  // - Partitions have latencies.
  // - Latency determines how many buffers the partition outputs need
  //   (SSA outputs).
  // - Latencies are assigned based on num_stages
}
