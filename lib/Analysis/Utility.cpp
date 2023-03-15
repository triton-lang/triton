#include "triton/Analysis/Utility.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <deque>

namespace mlir {

bool ReduceOpHelper::isFastReduction() {
  auto srcLayout = srcTy.getEncoding();
  auto axis = op.getAxis();
  return axis == triton::gpu::getOrder(srcLayout)[0];
}

unsigned ReduceOpHelper::getInterWarpSize() {
  auto srcLayout = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto axis = op.getAxis();
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  unsigned sizeIntraWarps = getIntraWarpSize();
  return std::min(srcReduceDimSize / sizeIntraWarps,
                  triton::gpu::getWarpsPerCTA(srcLayout)[axis]);
}

unsigned ReduceOpHelper::getIntraWarpSize() {
  auto srcLayout = srcTy.getEncoding();
  auto srcShape = srcTy.getShape();
  auto axis = op.getAxis();
  auto srcReduceDimSize = static_cast<unsigned>(srcShape[axis]);
  return std::min(srcReduceDimSize,
                  triton::gpu::getThreadsPerWarp(srcLayout)[axis]);
}

unsigned ReduceOpHelper::getThreadsReductionAxis() {
  auto srcLayout = srcTy.getEncoding();
  auto axis = op.getAxis();
  return triton::gpu::getThreadsPerWarp(srcLayout)[axis] *
         triton::gpu::getWarpsPerCTA(srcLayout)[axis];
}

SmallVector<unsigned> ReduceOpHelper::getScratchConfigBasic() {
  auto axis = op.getAxis();
  auto smemShape = convertType<unsigned>(getSrcShape());
  smemShape[axis] = std::min(smemShape[axis], getThreadsReductionAxis());
  return smemShape;
}

SmallVector<SmallVector<unsigned>> ReduceOpHelper::getScratchConfigsFast() {
  auto axis = op.getAxis();
  SmallVector<SmallVector<unsigned>> smemShapes(3);

  auto argLayout = srcTy.getEncoding();
  auto argLayoutMma = argLayout.dyn_cast<triton::gpu::MmaEncodingAttr>();
  if (argLayoutMma && argLayoutMma.getVersionMajor() == 2 &&
      triton::gpu::getWarpsPerCTA(argLayout)[axis] == 1)
    return {{1, 1}, {1, 1}};

  /// shared memory block0
  smemShapes[0] = convertType<unsigned>(getSrcShape());
  smemShapes[0][axis] = getInterWarpSize();

  /// FIXME(Qingyi): This size is actually larger than required.
  /// shared memory block1:
  auto mod = op.getOperation()->getParentOfType<ModuleOp>();
  unsigned numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);
  smemShapes[1].push_back(numWarps * 32);

  return smemShapes;
}

unsigned ReduceOpHelper::getScratchSizeInBytes() {
  unsigned elems = 0;
  if (isFastReduction()) {
    auto smemShapes = getScratchConfigsFast();
    for (const auto &smemShape : smemShapes)
      elems = std::max(elems, product<unsigned>(smemShape));
  } else {
    auto smemShape = getScratchConfigBasic();
    elems = product<unsigned>(smemShape);
  }

  auto tensorType = op.getOperand().getType().cast<RankedTensorType>();
  unsigned bytes = elems * tensorType.getElementTypeBitWidth() / 8;

  if (triton::ReduceOp::withIndex(op.getRedOp()))
    bytes += elems * sizeof(int32_t);

  return bytes;
}

bool isSharedEncoding(Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>()) {
    auto encoding = tensorType.getEncoding();
    return encoding && encoding.isa<triton::gpu::SharedEncodingAttr>();
  }
  return false;
}

bool maybeSharedAllocationOp(Operation *op) {
  // TODO(Keren): This function can be replaced by adding
  // MemoryEffectOpInterface. We can then use the MemoryEffectOpInterface to
  // query the memory effects of the op.
  auto *dialect = op->getDialect();
  return dialect &&
         (dialect->getTypeID() ==
              mlir::TypeID::get<triton::gpu::TritonGPUDialect>() ||
          dialect->getTypeID() == mlir::TypeID::get<triton::TritonDialect>() ||
          dialect->getTypeID() == mlir::TypeID::get<arith::ArithDialect>() ||
          dialect->getTypeID() == mlir::TypeID::get<tensor::TensorDialect>());
}

bool maybeAliasOp(Operation *op) {
  return isa<triton::gpu::ExtractSliceOp>(op) || isa<triton::TransOp>(op) ||
         isa<triton::gpu::InsertSliceAsyncOp>(op) ||
         isa<tensor::InsertSliceOp>(op);
}

bool supportMMA(triton::DotOp op, int version) {
  // Refer to mma section for the data type supported by Volta and Hopper
  // Tensor Core in
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
  auto aElemTy = op.getA().getType().cast<RankedTensorType>().getElementType();
  auto bElemTy = op.getB().getType().cast<RankedTensorType>().getElementType();
  if (aElemTy.isF32() && bElemTy.isF32()) {
    return op.getAllowTF32() && version >= 2;
  }
  return supportMMA(op.getA(), version) && supportMMA(op.getB(), version);
}

bool supportMMA(Value value, int version) {
  // Tell whether a DotOp support HMMA by the operand type(either $a or $b).
  // We cannot get both the operand types(in TypeConverter), here we assume the
  // types of both the operands are identical here.
  assert((version == 1 || version == 2) &&
         "Unexpected MMA layout version found");
  auto elemTy = value.getType().cast<RankedTensorType>().getElementType();
  return elemTy.isF16() || elemTy.isBF16() ||
         (elemTy.isF32() && version >= 2) ||
         (elemTy.isInteger(8) && version >= 2);
}

Type getElementType(Value value) {
  auto type = value.getType();
  if (auto tensorType = type.dyn_cast<RankedTensorType>())
    return tensorType.getElementType();
  return type;
}

std::string getValueOperandName(Value value, AsmState &state) {
  std::string opName;
  llvm::raw_string_ostream ss(opName);
  value.printAsOperand(ss, state);
  return opName;
}

bool isMmaToDotShortcut(triton::gpu::MmaEncodingAttr &mmaLayout,
                        triton::gpu::DotOperandEncodingAttr &dotOperandLayout) {
  // dot_op<opIdx=0, parent=#mma> = #mma
  // when #mma = MmaEncoding<version=2, warpsPerCTA=[..., 1]>
  return mmaLayout.getVersionMajor() == 2 &&
         mmaLayout.getWarpsPerCTA()[1] == 1 &&
         dotOperandLayout.getOpIdx() == 0 &&
         dotOperandLayout.getParent() == mmaLayout;
}

bool isSingleValue(Value value) {
  // Don't consider load as expensive if it is loading a scalar.
  if (auto tensorTy = value.getType().dyn_cast<RankedTensorType>())
    return tensorTy.getNumElements() == 1;
  // TODO: Handle other cases.
  // For example, when ptr is a tensor of single value.
  // It means that ptr is a resultant of broadcast or generated through
  // a chain of broadcast and other operations.
  // Rematerialize it without considering contiguous memory access pattern is
  // fine.
  return true;
}

namespace {

/// A data structure similar to SetVector but maintains
/// a deque instead of a vector to allow for efficient
/// push_back and pop_front operations.
/// Using SetVector doesn't suffice our needs because
/// it only pushes and pops from the back.
/// For example, if we have a queue like this:
/// 0->4 1->2->3
///    ^--------
/// where 3 depends on 4, once we pop 3, we found
/// 4 is not ready, so we check 2 and push 3 back
/// to the queue.
struct DFSSubgraphState {
  DFSSubgraphState() : set(), deque() {}
  DenseSet<Operation *> set;
  std::deque<Operation *> deque;

  bool push_back(Operation *op) {
    if (set.insert(op).second) {
      deque.push_back(op);
      return true;
    }
    return false;
  }

  Operation *pop_front() {
    Operation *op = deque.front();
    deque.pop_front();
    set.erase(op);
    return op;
  }

  bool empty() { return deque.empty(); }
};

/// DFS post-order implementation that maintains a global count to work across
/// multiple invocations, to help implement topological sort on multi-root DAGs.
/// We traverse all operations but only record the ones that appear in
/// `toSort` for the final result.
struct DFSState {
  DFSState(const SetVector<Operation *> &set) : toSort(set), seen() {}
  const SetVector<Operation *> &toSort;
  SmallVector<Operation *, 16> topologicalCounts;
  DenseSet<Operation *> seen;

  /// We mark each op as ready if all its operands are seen. If an op is ready,
  /// we add it to the queue. Otherwise, we keep adding its operands to the
  /// ancestors set.
  void addToReadyQueue(Operation *op, DFSSubgraphState &subGraph,
                       SmallVector<Operation *, 4> &readyQueue) {
    bool ready = true;
    for (Value operand : op->getOperands()) {
      auto def = operand.getDefiningOp();
      if (def && !seen.count(def)) {
        subGraph.push_back(def);
        ready = false;
      }
    }
    if (ready)
      readyQueue.push_back(op);
  }
};

void dfsPostorder(Operation *root, DFSState *state) {
  DFSSubgraphState subGraph;
  subGraph.push_back(root);
  SmallVector<Operation *> ops;
  while (!subGraph.empty()) {
    // Nodes in the ready queue are ready to be processed.
    // Meaning that either their operands are all seen or they have null
    // operands.
    SmallVector<Operation *, 4> readyQueue;
    auto *current = subGraph.pop_front();
    state->addToReadyQueue(current, subGraph, readyQueue);
    while (!readyQueue.empty()) {
      Operation *current = readyQueue.pop_back_val();
      if (!state->seen.insert(current).second)
        continue;
      ops.push_back(current);
      for (Value result : current->getResults()) {
        for (Operation *op : result.getUsers())
          state->addToReadyQueue(op, subGraph, readyQueue);
      }
      for (Region &region : current->getRegions()) {
        for (Operation &op : region.getOps())
          state->addToReadyQueue(&op, subGraph, readyQueue);
      }
    }
  }

  for (Operation *op : llvm::reverse(ops)) {
    if (state->toSort.count(op) > 0)
      state->topologicalCounts.push_back(op);
  }
}

} // namespace

SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort) {
  if (toSort.empty()) {
    return toSort;
  }

  // Run from each root with global count and `seen` set.
  DFSState state(toSort);
  for (auto *s : toSort) {
    assert(toSort.count(s) == 1 && "NYI: multi-sets not supported");
    dfsPostorder(s, &state);
  }

  // Reorder and return.
  SetVector<Operation *> res;
  for (auto it = state.topologicalCounts.rbegin(),
            eit = state.topologicalCounts.rend();
       it != eit; ++it) {
    res.insert(*it);
  }
  return res;
}

SetVector<Operation *> multiRootGetSlice(Operation *op,
                                         TransitiveFilter backwardFilter,
                                         TransitiveFilter forwardFilter) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    getBackwardSlice(currentOp, &backwardSlice, backwardFilter);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return multiRootTopologicalSort(slice);
}

namespace {
// Copied from TestDeadCodeAnalysis.cpp, because some dead code analysis
// interacts with constant propagation, but SparseConstantPropagation
// doesn't seem to be sufficient.
class ConstantAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override {
    WalkResult result = top->walk([&](Operation *op) {
      if (failed(visit(op)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  LogicalResult visit(ProgramPoint point) override {
    Operation *op = point.get<Operation *>();
    Attribute value;
    if (matchPattern(op, m_Constant(&value))) {
      auto *constant = getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(
          op->getResult(0));
      propagateIfChanged(constant, constant->join(dataflow::ConstantValue(
                                       value, op->getDialect())));
      return success();
    }
    // Dead code analysis requires every operands has initialized ConstantValue
    // state before it is visited.
    // https://github.com/llvm/llvm-project/blob/2ec1aba2b69faa1de5f71832a48e25aa3b5d5314/mlir/lib/Analysis/DataFlow/DeadCodeAnalysis.cpp#L322
    // That's why we need to set all operands to unknown constants.
    setAllToUnknownConstants(op->getResults());
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks())
        setAllToUnknownConstants(block.getArguments());
    }
    return success();
  }

private:
  /// Set all given values as not constants.
  void setAllToUnknownConstants(ValueRange values) {
    dataflow::ConstantValue unknownConstant(nullptr, nullptr);
    for (Value value : values) {
      auto *constant =
          getOrCreate<dataflow::Lattice<dataflow::ConstantValue>>(value);
      propagateIfChanged(constant, constant->join(unknownConstant));
    }
  }
};
} // namespace

std::unique_ptr<DataFlowSolver> createDataFlowSolver() {
  auto solver = std::make_unique<DataFlowSolver>();
  solver->load<dataflow::DeadCodeAnalysis>();
  solver->load<ConstantAnalysis>();
  return solver;
}

} // namespace mlir
