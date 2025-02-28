#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

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
  LogicalResult run();

private:
  // The schedule to apply.
  const WarpSchedule &schedule;
  // A precomputed partition graph.
  const PartitionGraph &graph;
  // The loop to partition.
  scf::ForOp loop;
};

struct Multiplicity {
  struct Node {
    unsigned argIdx = -1;
    unsigned depth = 0;
    Node *parent = nullptr;
    llvm::SmallSetVector<Node *, 2> children;
    int number = -1;
  };

  Node *getOrCreate(unsigned idx, unsigned depth, Node *parent);
  void number();
  unsigned getTotalBranchDepth();

  std::unique_ptr<Node> root = std::make_unique<Node>();
  llvm::MapVector<unsigned, std::unique_ptr<Node>> nodes;
  SmallVector<std::pair<unsigned, unsigned>> segments;
};

struct UseInfo {
  llvm::MapVector<std::pair<const Partition *, unsigned>,
                  SmallVector<OpOperand *>>
      consumers;
  Multiplicity multiplicity;
};
} // namespace

Multiplicity::Node *Multiplicity::getOrCreate(unsigned idx, unsigned depth,
                                              Node *parent) {
  std::unique_ptr<Node> &node = nodes[idx];
  if (!node) {
    node = std::make_unique<Node>();
    node->argIdx = idx;
    node->depth = depth;
    node->parent = parent;
  }
  assert(node->depth == depth && node->parent == parent && "conflicting node");
  return node.get();
}

void Multiplicity::number() {
  SmallVector<Node *> dfs;
  dfs.push_back(root.get());
  int number = 0;
  unsigned segmentStart = 0;
  while (!dfs.empty()) {
    Node *node = dfs.pop_back_val();
    node->number = number;
    llvm::append_range(dfs, node->children);
    if (node->children.empty()) {
      segments.emplace_back(segmentStart, segmentStart + node->depth);
      segmentStart += node->depth;
      ++number;
    }
  }
}

unsigned Multiplicity::getTotalBranchDepth() {
  unsigned multiplicitySize = 0;
  for (Multiplicity::Node &node :
       llvm::make_pointee_range(llvm::make_second_range(nodes))) {
    if (node.children.empty())
      multiplicitySize += node.depth;
  }
  return multiplicitySize;
}

// A partition output used a future iteration could get carried as an iter arg
// that is used in two different partitions. Consider:
//
//   scf.for %i = %lb to %ub step %step (%arg = %init)
//     %next = op_c()   {ttg.partition = 0}
//     op_a(%arg)       {ttg.partition = 1}
//     op_b(%arg)       {ttg.partition = 2}
//     scf.yield %next
//
// Output `%next` is used in partitions #1 and #2 but through the same arg.
static void resolveMultiplicity(ArrayRef<OpOperand *> uses, unsigned distance,
                                Multiplicity &mp, scf::YieldOp yield) {
  for (OpOperand *use : uses) {
    OpOperand *curUse = use;
    SmallVector<unsigned> trace;
    for (unsigned d = distance; d; --d) {
      auto arg = cast<BlockArgument>(curUse->get());
      unsigned idx = arg.getArgNumber() - 1;
      trace.push_back(idx);
      curUse = &yield.getResultsMutable()[idx];
    }
    Multiplicity::Node *parent = mp.root.get();
    for (auto [depth, idx] : llvm::enumerate(llvm::reverse(trace))) {
      Multiplicity::Node *node = mp.getOrCreate(idx, depth + 1, parent);
      parent->children.insert(node);
      parent = node;
    }
  }
}

static void
resolveOutputMultiplicity(llvm::MapVector<OpResult, UseInfo> &useInfo,
                          const Partition &partition, scf::YieldOp yield) {
  for (UseInfo &info : llvm::make_second_range(useInfo)) {
    for (auto [key, uses] : info.consumers) {
      auto [usePartition, distance] = key;
      if (usePartition == &partition) {
        // A partition using its own output won't be split across partitions.
        assert(distance > 0 && "self recursion must occur in the future");
        continue;
      }
      if (distance == 0) {
        // This is a use of the output in the current iteration.
        continue;
      }
      // We have uses of a value in a future iteration.
      resolveMultiplicity(uses, distance, info.multiplicity, yield);
    }
  }
}

static OpOperand *getEarliestUse(ArrayRef<OpOperand *> uses) {
  return *llvm::min_element(uses, [](OpOperand *lhs, OpOperand *rhs) {
    return lhs->getOwner()->isBeforeInBlock(rhs->getOwner());
  });
}

static Value i8cst(ImplicitLocOpBuilder &b, int value) {
  return b.create<arith::ConstantOp>(b.getI8IntegerAttr(value));
}

LogicalResult LoopPartitioner::run() {
  SmallVector<llvm::MapVector<OpResult, UseInfo>> partitionUseInfo;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (const Partition &partition : schedule.getPartitions()) {
    // Find all consumers of all outputs of this partition, tracking the
    // specific partition and distance of each use.
    auto &useInfo = partitionUseInfo.emplace_back();
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      const Partition *usePartition = schedule.getPartition(use.getOwner());
      UseInfo &info = useInfo[output];
      info.consumers[{usePartition, distance}].push_back(&use);
    };
    schedule.iterateUses(loop, &partition, callback);
    resolveOutputMultiplicity(useInfo, partition, yield);
  }

  // Cut all SSA dependencies by passing outputs through shared memory.
  ImplicitLocOpBuilder b(loop.getLoc(), loop);
  Value one = i8cst(b, 1);
  for (auto [partition, useInfo] :
       llvm::zip(schedule.getPartitions(), partitionUseInfo)) {
    // The amount of buffering is based on the longest distance to a user.
    for (auto &[output, info] : useInfo) {
      int maxDistance = 0;
      for (auto [usePartition, distance] :
           llvm::make_first_range(info.consumers)) {
        int dist = usePartition->getStage() - partition.getStage() + distance;
        assert(dist > 0 && "expected verifier to check schedule validity");
        maxDistance = std::max(maxDistance, dist);
      }
      if (!isa<IntegerType, FloatType>(output.getType())) {
        return mlir::emitError(output.getLoc(),
                               "FIXME: only integers and floats can be passed "
                               "through shared memory");
      }

      // FIXME(jeff): Factor this code out...
      b.setLoc(output.getLoc());
      int multiplicitySize = info.multiplicity.getTotalBranchDepth();
      int numBars = maxDistance + multiplicitySize;
      Value alloc = createScalarAlloc(b, output.getType(), numBars);
      Value readyBars = createScalarAlloc(b, b.getI64Type(), numBars);
      Value emptyBars = createScalarAlloc(b, b.getI64Type(), numBars);

      Block *body = loop.getBody();
      for (auto &[key, uses] : info.consumers) {
        assert(!uses.empty() && "expected at least one use");

        auto [usePartition, distance] = key;
        Value idx = body->addArgument(b.getI8Type(), b.getLoc());
        Value phase = body->addArgument(b.getI8Type(), b.getLoc());
        OpOperand *earliestUse = getEarliestUse(uses);
        b.setInsertionPointToStart(body);
        idx = b.create<arith::AddIOp>(idx, one);
        Value nextPhase = b.create<arith::XOrIOp>(phase, one);
        Value cnd = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, idx,
                                            i8cst(b, numBars));
        int startIdx = multiplicitySize;
        if (distance != 0) {
          unsigned argIdx =
              cast<BlockArgument>(uses.front()->get()).getArgNumber() - 1;
          Multiplicity::Node *node =
              info.multiplicity.nodes.find(argIdx)->second.get();
          auto [start, end] = info.multiplicity.segments[node->number];
          startIdx = start;
          if (end != multiplicitySize) {
            Value initEnd = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                    idx, i8cst(b, end));
            cnd = b.create<arith::AndIOp>(cnd, initEnd);
          }
        }

        idx = b.create<arith::SelectOp>(cnd, i8cst(b, multiplicitySize), idx);
        phase = b.create<arith::SelectOp>(cnd, nextPhase, phase);
      }
    }
  }

  llvm::SmallSetVector<const Partition *, 4> splitOrder;
  for (auto it = llvm::scc_begin(graph); !it.isAtEnd(); ++it) {
    assert(!it.hasCycle() && "expected verifier to check this");
    splitOrder.insert(it->front().first->partition);
  }
  SmallVector<std::unique_ptr<Region>> partitionRegions;
  for (const Partition *partition : llvm::reverse(splitOrder)) {
    auto region = std::make_unique<Region>();
    Block &block = region->emplaceBlock();
    partitionRegions.push_back(std::move(region));

    for (Operation *op : partition->getOps())
      op->moveBefore(&block, block.end());
  }

  return success();
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
    if (failed(partitioner.run()))
      return signalPassFailure();
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
