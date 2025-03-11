#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Partition.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/WarpSpecialization.h"
#include "llvm/ADT/SCCIterator.h"

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

using Partition = WarpSchedule::Partition;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Get the earliest user of a value, assuming all users are in the same block.
static Operation *getEarliestUser(ArrayRef<OpOperand *> uses) {
  OpOperand *use = *llvm::min_element(uses, [](OpOperand *lhs, OpOperand *rhs) {
    return lhs->getOwner()->isBeforeInBlock(rhs->getOwner());
  });
  return use->getOwner();
}

// Create an i32 constant.
static Value intCst(ImplicitLocOpBuilder &b, int value) {
  return b.create<arith::ConstantOp>(b.getI32IntegerAttr(value));
}

// Create an operation inside a partition.
template <typename OpT, typename... Args>
static auto createInPartition(ImplicitLocOpBuilder &b, Partition &partition,
                              Args &&...args) {
  auto op = b.create<OpT>(std::forward<Args>(args)...);
  partition.insert(op);
  return op;
}

// Get a generic shared encoding for a tensor.
static SharedEncodingTrait getSharedEncoding(RankedTensorType ty) {
  auto ctaLayout = getCTALayout(ty.getEncoding());
  auto order = getOrder(ty);
  // Use generic layout. This won't be optimal for 2D tensors.
  return SwizzledSharedEncodingAttr::get(ty.getContext(), 1, 1, 1, order,
                                         ctaLayout);
}

// Erase the given loop carried values from the loop, where `loop` is replaced
// with a new loop.
static void eraseLoopCarriedValues(scf::ForOp &loop, llvm::BitVector indices) {
  // Pad the indices in case new arguments were added.
  while (indices.size() != loop.getInitArgs().size())
    indices.push_back(false);

  loop.getBody()->getTerminator()->eraseOperands(indices);
  loop.getBody()->eraseArguments([&](BlockArgument arg) {
    int idx = arg.getArgNumber();
    return idx != 0 && indices.test(idx - 1);
  });

  llvm::BitVector loopOperandIndices(loop->getNumOperands());
  for (auto [i, operand] : llvm::enumerate(loop.getInitArgsMutable())) {
    if (indices.test(i))
      loopOperandIndices.set(operand.getOperandNumber());
  }
  loop->eraseOperands(loopOperandIndices);

  // Rewrite the loop to erase results.
  OperationState state(loop.getLoc(), loop->getName(), loop->getOperands(),
                       loop.getInitArgs().getTypes(), loop->getAttrs());
  state.addRegion()->takeBody(loop.getBodyRegion());

  OpBuilder b(loop);
  auto newLoop = cast<scf::ForOp>(b.create(state));
  loop->replaceAllUsesWith(
      newLoop.getResults().take_front(loop.getNumResults()));
  loop.erase();
  loop = newLoop;
}

//===----------------------------------------------------------------------===//
// Multiplicity
//===----------------------------------------------------------------------===//

namespace {
// Partition results are passed into future loop iterations through the region
// iter args of the loop. However, a specific block argument's SSA users could
// belong to multiple partitions or another `scf.yield`, where it is passed into
// another future iterations. Tracing forward the uses of a partition output
// creates a tree where each branch represents a series of block arguments and
// their corresponding initial values. Each use of the output in a future
// iteration is somewhere along the tree. To codegen the right initial values,
// the tree is flattened such that only the root node has multiple children.
//
// Consider:
//
//   for ... iter_args(%arg0 = %init0, %arg1 = %init1, %arg2 = %init2)
//     %output = f() {partition = 0}
//     use(%arg0)    {partition = 1}
//     use(%arg0)    {partition = 2}
//     use(%arg1)    {partition = 3}
//     use(%arg2)    {partition = 4}
//     yield %output, %arg0, %arg0
//
// The first iteration of partitions 1 and 2 must read `%init0` for the i+1
// value of `%output`, but partitions 2 and 3 read `%init1` and `%init2`
// respectively as the i+2 values of `%output`.
//
// The corresponding tree is:
//
//   %output (root) -> %arg0 -> %arg1
//                         \--> %arg2
//
struct Multiplicity {
  // A node in the multiplicity tree, representing a region iter arg.
  struct Node {
    // The index of the region iter arg.
    unsigned argIdx = -1;
    // The depth of the node in the tree, which actually represents the
    // distance.
    unsigned depth = 0;
    // The children nodes, i.e. the region iter args whose values are derived
    // from this iter arg.
    llvm::SmallSetVector<Node *, 2> children;

    // This value is set later but represents the branch index of the flattened
    // tree that this node belongs to.
    int number = -1;
  };

  // Get or create a node in the tree for the given region iter index and at a
  // certain depth (distance).
  Node *getOrCreate(unsigned idx, unsigned depth);
  // Add nodes to the multiplicity tree given uses of an output at a paritition
  // and distance.
  void add(ArrayRef<OpOperand *> uses, unsigned distance, scf::YieldOp yield);

  // The total branch depth is the number of nodes in the flattened tree, and
  // the number of extra buffers needed.
  unsigned getTotalBranchDepth();
  // Flatten the tree by assigning each node to a branch. A callback is invoked
  // for each branch and the start and end indices of the nodes along that
  // branch.
  void number(
      function_ref<void(ArrayRef<Node *>, unsigned, unsigned)> branchCallback);

  // The root node, which represents the partition output itself.
  std::unique_ptr<Node> root = std::make_unique<Node>();
  // All the nodes in the tree, mapped by region iter arg index.
  llvm::MapVector<unsigned, std::unique_ptr<Node>> nodes;
  // Map from branch index to buffer start and end index, populated by `number`.
  SmallVector<std::pair<unsigned, unsigned>> segments;
};
} // namespace

Multiplicity::Node *Multiplicity::getOrCreate(unsigned idx, unsigned depth) {
  std::unique_ptr<Node> &node = nodes[idx];
  if (!node) {
    node = std::make_unique<Node>();
    node->argIdx = idx;
    node->depth = depth;
  }
  // Check that the node is consistent.
  assert(node->depth == depth && "conflicting node");
  return node.get();
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
void Multiplicity::add(ArrayRef<OpOperand *> uses, unsigned distance,
                       scf::YieldOp yield) {
  for (OpOperand *use : uses) {
    OpOperand *curUse = use;
    SmallVector<unsigned> trace;
    for (unsigned d = distance; d; --d) {
      auto arg = cast<BlockArgument>(curUse->get());
      unsigned idx = arg.getArgNumber() - 1;
      trace.push_back(idx);
      curUse = &yield.getResultsMutable()[idx];
    }
    Multiplicity::Node *parent = root.get();
    for (auto [depth, idx] : llvm::enumerate(llvm::reverse(trace))) {
      Multiplicity::Node *node = getOrCreate(idx, depth + 1);
      parent->children.insert(node);
      parent = node;
    }
  }
}

void Multiplicity::number(
    function_ref<void(ArrayRef<Node *>, unsigned, unsigned)> branchCallback) {
  // Depth-first traversal of the tree. Each new branch is assigned to an
  // incremented branch index. Thus, the "flattened" tree is virtual and isn't
  // bigger than the actual tree.
  SmallVector<Node *> dfs;
  dfs.push_back(root.get());
  int number = 0;
  unsigned segmentStart = 0;
  // Keep the traceback of the nodes along the virtual branch.
  SmallVector<Node *> trace;
  while (!dfs.empty()) {
    Node *node = dfs.pop_back_val();
    trace.push_back(node);
    // Assign the branch index.
    node->number = number;
    llvm::append_range(dfs, node->children);

    // If there are no children, we know we reached the end of a branch.
    if (node->children.empty()) {
      // Save the segment indices.
      segments.emplace_back(segmentStart, segmentStart + node->depth);
      auto [start, end] = segments.back();
      assert(node->depth == trace.size() - 1);
      // Don't include the root.
      branchCallback(ArrayRef(trace).drop_front(), start, end);

      // Move to the next branch.
      segmentStart += node->depth;
      ++number;
      if (!dfs.empty())
        trace.resize(dfs.back()->depth - 1);
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

//===----------------------------------------------------------------------===//
// DependencyRewriter
//===----------------------------------------------------------------------===//

namespace {
// Helper class for dependency rewriting.
class DependencyRewriter {
public:
  DependencyRewriter(WarpSchedule &schedule, scf::ForOp loop)
      : schedule(schedule), loop(loop) {}

  // Partition the loop.
  LogicalResult run();

private:
  // The schedule to apply.
  WarpSchedule &schedule;
  // The loop to partition.
  scf::ForOp loop;
};

// Use information for a partition SSA output.
struct UseInfo {
  // Map from partition and distance to the uses in that partition.
  llvm::MapVector<std::pair<Partition *, unsigned>, SmallVector<OpOperand *>>
      consumers;
  // The multiplicity tree of the output.
  Multiplicity multiplicity;
};
} // namespace

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
      info.multiplicity.add(uses, distance, yield);
    }
  }
}

LogicalResult DependencyRewriter::run() {
  SmallVector<llvm::MapVector<OpResult, UseInfo>> partitionUseInfo;
  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (const Partition &partition : schedule.getPartitions()) {
    // Find all consumers of all outputs of this partition, tracking the
    // specific partition and distance of each use.
    auto &useInfo = partitionUseInfo.emplace_back();
    auto callback = [&](OpResult output, OpOperand &use, unsigned distance) {
      Partition *usePartition = schedule.getPartition(use.getOwner());
      UseInfo &info = useInfo[output];
      info.consumers[{usePartition, distance}].push_back(&use);
    };
    schedule.iterateUses(loop, &partition, callback);
    resolveOutputMultiplicity(useInfo, partition, yield);
  }

  // Cut all SSA dependencies by passing outputs through shared memory.
  ImplicitLocOpBuilder b(loop.getLoc(), loop);
  Value zero = intCst(b, 0);
  Value one = intCst(b, 1);
  llvm::BitVector toErase(yield.getNumOperands());
  for (auto [partition, useInfo] :
       llvm::zip(schedule.getPartitions(), partitionUseInfo)) {
    // The amount of buffering is based on the longest distance to a user.
    for (auto &[output, info] : useInfo) {
      // Buffer the value based on the greatest distance to a consumer
      // partition.
      int maxDistance = 0;
      for (auto [usePartition, distance] :
           llvm::make_first_range(info.consumers)) {
        int dist = usePartition->getStage() - partition.getStage() + distance;
        assert(dist > 0 && "expected verifier to check schedule validity");
        maxDistance = std::max(maxDistance, dist);
      }

      // FIXME: No IR support for passing simple scalars through shared memory.
      auto tensorType = dyn_cast<RankedTensorType>(output.getType());
      if (!tensorType) {
        return mlir::emitWarning(output.getLoc(),
                               "FIXME: only tensor SSA dependencies between "
                               "partitions are supported");
      }

      // Number the branches of the multiplicity tree. The total number of
      // required buffers includes the lengths of all branches.
      int multiplicitySize = info.multiplicity.getTotalBranchDepth();
      int numBars = maxDistance + multiplicitySize;

      // Allocate buffers for the value and its associated barriers.
      b.setLoc(output.getLoc());
      Value alloc = createAlloc(loop, tensorType, output.getLoc(),
                                getSharedEncoding(tensorType), numBars);
      Value readyBars = createScalarAlloc(b, b.getI64Type(), numBars);
      Value emptyBars = createScalarAlloc(b, b.getI64Type(), numBars);

      auto allocType = cast<MemDescType>(alloc.getType());
      MemDescType viewType = getBufferViewType(allocType);

      // Initialize the initial values of the loop-carried dependencies.
      unsigned numConsumers = info.consumers.size();
      ImplicitLocOpBuilder endBuilder(b.getLoc(), loop->getNextNode());
      info.multiplicity.number([&](ArrayRef<Multiplicity::Node *> nodes,
                                   unsigned start, unsigned end) {
        for (auto [i, node] : llvm::zip(llvm::seq(start, end), nodes)) {
          Value idx = intCst(b, i);
          SmallVector<Value> offsets(allocType.getRank(), zero);
          offsets.front() = idx;
          Value view = b.create<MemDescSubviewOp>(viewType, alloc, offsets);
          Value readyView = createSingleBufferView(b, readyBars, idx);
          Value emptyView = createSingleBufferView(b, emptyBars, idx);
          b.create<LocalStoreOp>(loop.getInitArgs()[node->argIdx], view);
          b.create<ttng::InitBarrierOp>(readyView, 1);
          b.create<ttng::InitBarrierOp>(emptyView, numConsumers);
          b.create<ttng::ArriveBarrierOp>(readyView, 1);
          endBuilder.create<ttng::InvalBarrierOp>(readyView);
          endBuilder.create<ttng::InvalBarrierOp>(emptyView);
        }
      });

      // Initialize the buffers.
      for (auto i : llvm::seq(multiplicitySize, numBars)) {
        Value idx = intCst(b, i);
        SmallVector<Value> offsets(allocType.getRank(), zero);
        offsets.front() = idx;
        Value readyView = createSingleBufferView(b, readyBars, idx);
        Value emptyView = createSingleBufferView(b, emptyBars, idx);
        b.create<ttng::InitBarrierOp>(readyView, 1);
        b.create<ttng::InitBarrierOp>(emptyView, numConsumers);
        b.create<ttng::ArriveBarrierOp>(emptyView, numConsumers);
        endBuilder.create<ttng::InvalBarrierOp>(readyView);
        endBuilder.create<ttng::InvalBarrierOp>(emptyView);
      }
      endBuilder.create<LocalDeallocOp>(readyBars);
      endBuilder.create<LocalDeallocOp>(emptyBars);

      Block *body = loop.getBody();
      for (auto &[key, uses] : info.consumers) {
        assert(!uses.empty() && "expected at least one use");

        auto [usePartition, distance] = key;
        // Thread the phase and buffer index through the loop. The index is
        // pre-incremented.
        Value idx = body->addArgument(b.getI32Type(), b.getLoc());
        Value phase = body->addArgument(b.getI32Type(), b.getLoc());
        Operation *earliestUser = getEarliestUser(uses);
        b.setInsertionPoint(earliestUser);
        idx = b.create<arith::AddIOp>(idx, one);
        Value nextPhase = b.create<arith::XOrIOp>(phase, one);
        Value cnd = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, idx,
                                            intCst(b, numBars));
        // The phase flips when we reach the end of all buffers.
        phase = b.create<arith::SelectOp>(cnd, nextPhase, phase);

        // Additionally rollover the buffer index when it has multiplicity.
        int startIdx = multiplicitySize;
        if (distance != 0) {
          unsigned argIdx =
              cast<BlockArgument>(uses.front()->get()).getArgNumber() - 1;
          toErase.set(argIdx);
          Multiplicity::Node *node =
              info.multiplicity.nodes.find(argIdx)->second.get();
          auto [start, end] = info.multiplicity.segments[node->number];
          startIdx = end - node->depth;
          assert(node->depth && startIdx >= start && "incorrect numbering?");
          // Micro-optimization: if the index would roll onto `multiplicitySize`
          // anyways, skip generating the check.
          if (end != multiplicitySize) {
            Value initEnd = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                    idx, intCst(b, end));
            cnd = b.create<arith::AndIOp>(cnd, initEnd);
          }
        }

        idx = b.create<arith::SelectOp>(cnd, intCst(b, multiplicitySize), idx);
        yield.getResultsMutable().append({idx, phase});
        {
          OpBuilder::InsertionGuard guard(b);
          b.setInsertionPoint(loop);
          // The index is preincremented so subtract 1 from the start.
          loop.getInitArgsMutable().append({intCst(b, startIdx - 1), zero});
        }

        SmallVector<Value> offsets(allocType.getRank(), zero);
        offsets.front() = idx;
        Value view = b.create<MemDescSubviewOp>(viewType, alloc, offsets);
        Value readyView = createSingleBufferView(b, readyBars, idx);
        Value emptyView = createSingleBufferView(b, emptyBars, idx);

        // Up until here, the ops we created are trivial arithmetic and can
        // simply live in the root partition. The next ops must be assigned to
        // the right partition.

        // Wait for the value to be available.
        createInPartition<ttng::WaitBarrierOp>(b, *usePartition, readyView,
                                               phase);
        // Load the value at the current index and replace uses in this
        // partition with it.
        Value value =
            createInPartition<LocalLoadOp>(b, *usePartition, tensorType, view);
        for (OpOperand *use : uses)
          use->set(value);
        // Mark the buffer as ready.
        createInPartition<ttng::ArriveBarrierOp>(b, *usePartition, emptyView,
                                                 1);
      }

      // Set up production of the value.
      b.setInsertionPointAfter(output.getDefiningOp());
      Value idx = body->addArgument(b.getI32Type(), b.getLoc());
      Value phase = body->addArgument(b.getI32Type(), b.getLoc());
      idx = b.create<arith::AddIOp>(idx, one);
      Value nextPhase = b.create<arith::XOrIOp>(phase, one);
      Value cnd = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq, idx,
                                          intCst(b, numBars));
      phase = b.create<arith::SelectOp>(cnd, nextPhase, phase);
      idx = b.create<arith::SelectOp>(cnd, intCst(b, multiplicitySize), idx);
      yield.getResultsMutable().append({idx, phase});
      {
        OpBuilder::InsertionGuard guard(b);
        b.setInsertionPoint(loop);
        loop.getInitArgsMutable().append(
            {intCst(b, multiplicitySize - 1), zero});
      }

      SmallVector<Value> offsets(allocType.getRank(), zero);
      offsets.front() = idx;
      Value view = b.create<MemDescSubviewOp>(viewType, alloc, offsets);
      Value readyView = createSingleBufferView(b, readyBars, idx);
      Value emptyView = createSingleBufferView(b, emptyBars, idx);

      createInPartition<ttng::WaitBarrierOp>(b, partition, emptyView, phase);
      createInPartition<LocalStoreOp>(b, partition, output, view);
      createInPartition<ttng::ArriveBarrierOp>(b, partition, readyView, 1);
    }
  }

  // Erase unneeded loop-carried values.
  eraseLoopCarriedValues(loop, std::move(toErase));

  // Update the schedule.
  schedule.serialize(loop);
  return success();
}

//===----------------------------------------------------------------------===//
// rewritePartitionDependenies
//===----------------------------------------------------------------------===//

LogicalResult triton::gpu::rewritePartitionDependencies(scf::ForOp loop) {
  FailureOr<WarpSchedule> scheduleOr = WarpSchedule::deserialize(loop);
  if (failed(scheduleOr))
    return failure();
  WarpSchedule schedule = std::move(*scheduleOr);
  if (failed(schedule.verify(loop)))
    return failure();
  DependencyRewriter rewriter(schedule, loop);
  if (failed(rewriter.run()))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace mlir::triton::gpu {
#define GEN_PASS_DEF_TRITONGPUREWRITEPARTITIONDEPENDENCIES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"
} // namespace mlir::triton::gpu

namespace {
struct RewritePartitionDependencies
    : triton::gpu::impl::TritonGPURewritePartitionDependenciesBase<
          RewritePartitionDependencies> {
            using TritonGPURewritePartitionDependenciesBase::TritonGPURewritePartitionDependenciesBase;

  void runOnOperation() override;
};
} // namespace

void RewritePartitionDependencies::runOnOperation() {
  // Collect for loops to warp specialize. This pass expects the loop to already
  // be scheduled.
  SmallVector<scf::ForOp> loops;
  getOperation().walk([&](scf::ForOp loop) {
    if (loop->hasAttrOfType<ArrayAttr>(kPartitionStagesAttrName))
      loops.push_back(loop);
  });

  for (scf::ForOp loop : loops) {
    if (failed(rewritePartitionDependencies(loop)))
      continue;
  }
}
