#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"
#include <map>

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {
namespace {
struct SliceNode {
  Operation *op;
  SmallVector<SliceNode *> successors;
  unsigned component = ~0u;
};

struct SliceGraphTraits {
  using NodeRef = SliceNode *;
  using ChildIteratorType = SmallVectorImpl<SliceNode *>::iterator;
  static NodeRef getEntryNode(NodeRef node) { return node; }
  static ChildIteratorType child_begin(NodeRef node) {
    return node->successors.begin();
  }
  static ChildIteratorType child_end(NodeRef node) {
    return node->successors.end();
  }
};

using ShapeAndOrder = std::pair<SmallVector<int64_t>, SmallVector<unsigned>>;
using VectorizationMap = std::map<ShapeAndOrder, unsigned>;
} // namespace

void CoalescedEncodingAnalysis::computeSliceInfo(FunctionOpInterface func) {
  SmallVector<SliceNode> nodes;
  DenseMap<Operation *, unsigned> nodeIndices;
  func.walk([&](Operation *op) {
    nodeIndices[op] = nodes.size();
    nodes.push_back({op, {}});
  });
  auto getNode = [&](Operation *op) {
    auto it = nodeIndices.find(op);
    assert(it != nodeIndices.end() && "slice must stay within the function");
    return &nodes[it->second];
  };

  // These are the immediate edges followed by getSlice with default options.
  // Walking them once avoids repeatedly computing transitive forward/backward
  // slices. Region containment is directed: a parent reaches its nested ops,
  // but a nested op need not reach its parent.
  for (SliceNode &node : nodes) {
    Operation *op = node.op;
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        node.successors.push_back(getNode(def));
      } else {
        auto arg = cast<BlockArgument>(operand);
        Operation *parent = arg.getOwner()->getParentOp();
        if (parent && !parent->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
            parent->getNumRegions() == 1 && parent->getRegion(0).hasOneBlock())
          node.successors.push_back(getNode(parent));
      }
    }
    for (Operation *user : op->getUsers())
      node.successors.push_back(getNode(user));
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &nested : block)
          node.successors.push_back(getNode(&nested));
  }

  auto getKey = [&](Value ptr) -> ShapeAndOrder {
    auto type = cast<RankedTensorType>(ptr.getType());
    return {SmallVector<int64_t>(type.getShape()),
            getOrderFromContiguity(
                axisInfoAnalysis.getAxisInfo(ptr)->getContiguity())};
  };

  // Only operations in the same strongly connected component have identical
  // slices. SCCs are visited with successors first, so each component can reuse
  // the vectorization summaries of the components it reaches.
  SmallVector<VectorizationMap> summaries;
  using SCCIterator = llvm::scc_iterator<SliceNode *, SliceGraphTraits>;
  for (auto it = SCCIterator::begin(getNode(func)); !it.isAtEnd(); ++it) {
    unsigned component = summaries.size();
    auto &summary = summaries.emplace_back();
    for (SliceNode *node : *it)
      node->component = component;

    llvm::SmallDenseSet<unsigned> successors;
    for (SliceNode *node : *it) {
      for (SliceNode *successor : node->successors) {
        unsigned next = successor->component;
        if (next == component || !successors.insert(next).second)
          continue;
        for (const auto &[key, perThread] : summaries[next])
          summary[key] = std::max(summary[key], perThread);
      }
      Value ptr = getMemAccessPtr(node->op);
      if (!ptr || !isa<RankedTensorType>(ptr.getType()))
        continue;
      auto key = getKey(ptr);
      // The only dependence on the CTA shape is a minimum with its contiguous
      // extent. Aggregate using the full shape, then apply that minimum when
      // querying the layout, so different CTA layouts can share a summary.
      unsigned perThread = getNumElementsPerThread(node->op, key.second,
                                                   axisInfoAnalysis, key.first);
      summary[key] = std::max(summary[key], perThread);
    }
    for (SliceNode *node : *it) {
      if (!isa<LoadOp>(node->op))
        continue;
      Value ptr = getMemAccessPtr(node->op);
      if (isa<RankedTensorType>(ptr.getType()) && ptr.getDefiningOp())
        elementsPerThread[node->op] = summary.at(getKey(ptr));
    }
  }
}

BlockedEncodingAttr CoalescedEncodingAnalysis::getEncoding(
    Operation *op, int numWarps, int threadsPerWarp, CGAEncodingAttr cgaLayout,
    ArrayRef<int64_t> shapePerCTA) {
  Value ptr = getMemAccessPtr(op);
  auto refTensorType = cast<RankedTensorType>(ptr.getType());

  LDBG("Considering op: " << *op);
  LLVM_DEBUG({
    DBGS() << "axis info of pointer: ";
    axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
  SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
  LDBG("order=[" << triton::join(order, ", ") << "]");
  LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

  int numElems = product<int64_t>(shapePerCTA);
  int numThreads = numWarps * threadsPerWarp;
  unsigned perThread;
  if (isa<LoadOp>(op) && ptr.getDefiningOp()) {
    auto it = elementsPerThread.find(op);
    if (it == elementsPerThread.end()) {
      computeSliceInfo(op->getParentOfType<FunctionOpInterface>());
      it = elementsPerThread.find(op);
    }
    assert(it != elementsPerThread.end() && "expected a tensor load");
    perThread = std::min<int64_t>(it->second, shapePerCTA[order[0]]);
  } else {
    // Memory writes must retain their own vectorization bound to avoid gaps
    // between the writes of neighboring threads. The dependency maximum cannot
    // change this bound, so these operations need no slice analysis. Loads from
    // block-argument pointers also retain their original local bound.
    perThread =
        getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA);
  }
  perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
  LDBG("perThread: " << perThread);

  SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
  sizePerThread[order[0]] = perThread;
  return BlockedEncodingAttr::get(op->getContext(), refTensorType.getShape(),
                                  sizePerThread, order, numWarps,
                                  threadsPerWarp, cgaLayout);
}
} // namespace mlir::triton::gpu
