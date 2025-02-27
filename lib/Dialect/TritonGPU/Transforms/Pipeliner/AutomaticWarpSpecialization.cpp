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

//===----------------------------------------------------------------------===//
// Loop Partitioning
//===----------------------------------------------------------------------===//

namespace {
using Partition = WarpSchedule::Partition;

// Helper class for loop partitioning.
class LoopPartitioner {
public:
  LoopPartitioner(const WarpSchedule &schedule, const PartitionGraph &graph,
                  scf::ForOp loop)
      : schedule(schedule), graph(graph), loop(loop) {}

  // Partition the loop.
  void run();

private:
  // The schedule to apply.
  const WarpSchedule &schedule;
  // A precomputed partition graph.
  const PartitionGraph &graph;
  // The loop to partition.
  scf::ForOp loop;
};

struct UseInfo {
  llvm::MapVector<const Partition *, SmallVector<OpOperand *>> consumers;
  unsigned maxDistance = 0;
};
} // namespace

void LoopPartitioner::run() {
  for (const Partition &partition : schedule.getPartitions()) {
    // Find all the consumers of an output. That output will need to be
    // multibuffered based on the maximum total distance to its uses. At the
    // same time, because each consumer can complete at any time, track all of
    // them since they all need to be synchronized.
    DenseMap<OpResult, UseInfo> useInfo;

    unsigned defStage = partition.getStage();
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      // Determine the overall distance of the use. This is the stage delta plus
      // the distance in the future.
      const Partition *usePartition = schedule.getPartition(use.getOwner());
      unsigned useStage = usePartition->getStage();
      assert(useStage > defStage && "expected verifier to check this");

      UseInfo &info = useInfo[output];
      info.maxDistance = std::max(info.maxDistance, useStage - defStage);

      OpOperand *curUse = &use;
      while (distance--) {
        unsigned idx = cast<BlockArgument>(curUse->get()).getArgNumber();
        curUse = &(*loop.getYieldedValuesMutable())[idx - 1];
      }
      assert(curUse->getOwner() == loop.getBody()->getTerminator() &&
             curUse->get().getDefiningOp());
    };

    schedule.iterateUses(loop, &partition, callback);
  }
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

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
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(loop);
    if (failed(scheduleOr))
      continue;
    WarpSchedule schedule = std::move(*scheduleOr);
    FailureOr<PartitionGraph> graphOr = schedule.verify(loop);
    if (failed(graphOr))
      continue;
    LoopPartitioner partitioner(schedule, *graphOr, loop);
    partitioner.run();
  }

  // FIXME: Scratch notes below:

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
