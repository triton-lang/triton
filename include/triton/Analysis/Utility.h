#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>
#include <string>

namespace mlir {

class ReduceOpHelper {
public:
  explicit ReduceOpHelper(triton::ReduceOp op)
      : op(op.getOperation()), axis(op.getAxis()) {
    auto firstTy = op.getOperands()[0].getType().cast<RankedTensorType>();
    srcShape = firstTy.getShape();
    srcEncoding = firstTy.getEncoding();
    srcElementTypes = op.getElementTypes();

    for (const auto &t : op.getInputTypes()) {
      if (t.getShape() != srcShape) {
        op.emitError() << "shape mismatch";
      }
      if (t.getEncoding() != srcEncoding) {
        op.emitError() << "encoding mismatch";
      }
    }
  }

  ArrayRef<int64_t> getSrcShape() { return srcShape; }

  Attribute getSrcLayout() { return srcEncoding; }

  triton::ReduceOp getOperation() { return op; }

  bool isReductionOnLayoutFastAxis();

  unsigned getThreadOffsetOnReductionAxis();

  bool isWarpSynchronous();

  unsigned getInterWarpSize();

  unsigned getIntraWarpSize();

  unsigned getInterWarpSizeWithUniqueData();

  unsigned getIntraWarpSizeWithUniqueData();

  unsigned getThreadsReductionAxis();

  SmallVector<unsigned> getScratchConfig();

  SmallVector<unsigned> getOrderWithAxisAtBeginning();

  unsigned getScratchSizeInBytes();

  bool isSupportedLayout();

  bool isReduceWithinCTA();

  unsigned getAxis() { return axis; }

private:
  triton::ReduceOp op;
  ArrayRef<int64_t> srcShape;
  Attribute srcEncoding;
  SmallVector<Type> srcElementTypes;
  int axis;
};

class ScanLoweringHelper {
public:
  explicit ScanLoweringHelper(triton::ScanOp op) : scanOp(op) {
    auto type = scanOp.getOperand(0).getType().cast<RankedTensorType>();
    srcEncoding = type.getEncoding();
  }
  // Return true if the lowering of the scan op is supported.
  bool isSupported();
  // Return the number of elements per thread along axis dim.
  unsigned getAxisNumElementsPerThread();
  // Return the number of elements per thread along non-axis dims.
  unsigned getNonAxisNumElementsPerThread();
  // Return the number of threads per warp along non-axis dims.
  unsigned getNonAxisNumThreadsPerWarp();
  // Return the flat numbers of threads computing independent scan results.
  unsigned getNonAxisNumThreadsPerCTA();
  // Return the number of warps per CTA along axis dim.
  unsigned getAxisNumWarps();
  // Return the number of warps per CTA along axis dim with unique data.
  unsigned getAxisNumWarpsWithUniqueData();
  // Return the number of threads per warp along axis dim.
  unsigned getAxisNumThreadsPerWarp();
  // Return the number of threads per warp along axis dim with unique data.
  unsigned getAxisNumThreadsPerWarpWithUniqueData();
  // Return the number of blocks along axis dim.
  unsigned getAxisNumBlocks();
  // Return the number of blocks along non axis dim.
  unsigned getNonAxisNumBlocks();
  // Return the size of the scratch space needed for scan lowering.
  unsigned getScratchSizeInBytes();

  // Stride between contiguous element along axis dim.
  unsigned getAxisElementStride();
  // Stride between contiguous threads along axis dim.
  unsigned getAxisThreadStride();
  // Stride between contiguous blocks along axis dim.
  unsigned getAxisBlockStride();

  Location getLoc() { return scanOp.getLoc(); }
  unsigned getAxis() { return scanOp.getAxis(); }
  triton::gpu::BlockedEncodingAttr getEncoding();
  llvm::ArrayRef<int64_t> getShape();
  Region &getCombineOp();

private:
  triton::ScanOp scanOp;
  Attribute srcEncoding;
};

bool maybeSharedAllocationOp(Operation *op);

bool maybeAliasOp(Operation *op);

bool supportMMA(triton::DotOp op, int version);

bool supportMMA(Value value, int version);

bool isSingleValue(Value value);

bool isMmaToDotShortcut(RankedTensorType srcTy, RankedTensorType dstTy);

bool isMmaToMmaShortcut(RankedTensorType srcTy, RankedTensorType dstTy);

// Return true if the src and dst layout match.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy);

// TODO: Move utility functions that belong to ConvertLayoutOp to class
// ConvertLayoutOpHelper in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout);

template <typename T_OUT, typename T_IN>
inline SmallVector<T_OUT> convertType(ArrayRef<T_IN> in) {
  SmallVector<T_OUT> out;
  for (const T_IN &i : in)
    out.push_back(T_OUT(i));
  return out;
}

template <typename Int> Int product(llvm::ArrayRef<Int> arr) {
  return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies{});
}

template <typename Int> Int ceil(Int m, Int n) { return (m + n - 1) / n; }

/// output[i] = input[order[i]]
template <typename T, typename RES_T = T>
SmallVector<RES_T> reorder(ArrayRef<T> input, ArrayRef<unsigned> order) {
  size_t rank = order.size();
  assert(input.size() == rank);
  SmallVector<RES_T> result(rank);
  for (auto it : llvm::enumerate(order)) {
    result[it.index()] = input[it.value()];
  }
  return result;
}

/// Get the highest power of 2 divisor of an integer.
template <typename T> T highestPowOf2Divisor(T n) {
  if (n == 0) {
    return (static_cast<T>(1) << (sizeof(T) * 8 - 2));
  }
  return (n & (~(n - 1)));
}

/// Get the next power of 2 for an integer (or the integer itself if it is a
/// power of 2).
template <typename T> T nextPowOf2(T n) {
  if (n == 0) {
    return 1;
  }
  n--;
  for (unsigned i = 1; i < sizeof(T) * 8; i <<= 1) {
    n |= n >> i;
  }
  return n + 1;
}

/// Multi-root DAG topological sort.
/// Performs a topological sort of the Operation in the `toSort` SetVector.
/// Returns a topologically sorted SetVector.
/// It is faster than mlir::topologicalSort because it prunes nodes that have
/// been visited before.
SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort);

/// This uses the toplogicalSort above
SetVector<Operation *>
multiRootGetSlice(Operation *op, TransitiveFilter backwardFilter = nullptr,
                  TransitiveFilter forwardFilter = nullptr);

/// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

/// This class represents a call graph for a given ModuleOp and holds
/// data of type T associated with each FunctionOpInterface.
template <typename T> class CallGraph {
public:
  using FuncDataMapT = DenseMap<FunctionOpInterface, T>;

  /// Constructor that builds the call graph for the given moduleOp.
  explicit CallGraph(ModuleOp moduleOp) : moduleOp(moduleOp) { build(); }

  /// Walks the call graph and applies the provided update functions
  /// to the edges and nodes.
  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void walk(UpdateEdgeFn updateEdgeFn, UpdateNodeFn updateNodeFn) {
    DenseSet<FunctionOpInterface> visited;
    for (auto root : roots) {
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(root, visited, updateEdgeFn,
                                               updateNodeFn);
    }
  }

  /// Retrieves the data associated with a function
  T *getFuncData(FunctionOpInterface funcOp) {
    if (funcMap.count(funcOp)) {
      return &funcMap[funcOp];
    }
    return nullptr;
  }

  /// Getters
  ModuleOp getModuleOp() const { return moduleOp; }
  SmallVector<FunctionOpInterface> getRoots() const { return roots; }
  size_t getNumFunctions() const { return funcMap.size(); }

  /// Returns true if the given function is a root.
  bool isRoot(FunctionOpInterface funcOp) const {
    return llvm::is_contained(roots, funcOp);
  }

  /// Maps the data and the graph nodes associated with a funcOp to a
  /// targetFuncOp.
  template <typename FROM, typename TO>
  void mapFuncOp(FROM funcOp, TO targetFuncOp) {
    // Iterate over graph and replace
    for (auto &kv : graph) {
      for (auto &edge : kv.second) {
        if (edge.second == funcOp) {
          edge.second = targetFuncOp;
        }
      }
    }
    graph[targetFuncOp] = graph[funcOp];
    // Replace in roots
    for (auto it = roots.begin(); it != roots.end(); ++it) {
      if (*it == funcOp) {
        *it = targetFuncOp;
        break;
      }
    }
    // Replace in funcMap
    funcMap[targetFuncOp] = funcMap[funcOp];
  }

  /// Maps the graph edges associated with a callOp to a targetCallOp.
  template <typename FROM, typename TO>
  void mapCallOp(FROM callOp, TO targetCallOp) {
    // Iterate over graph and replace
    for (auto &kv : graph) {
      for (auto &edge : kv.second) {
        if (edge.first == callOp) {
          edge.first = targetCallOp;
        }
      }
    }
  }

private:
  void build() {
    SymbolTableCollection symbolTable;
    DenseSet<FunctionOpInterface> visited;
    // Build graph
    moduleOp.walk([&](Operation *op) {
      auto caller = op->getParentOfType<FunctionOpInterface>();
      if (auto callOp = dyn_cast<CallOpInterface>(op)) {
        auto *callee = callOp.resolveCallable(&symbolTable);
        auto funcOp = dyn_cast_or_null<FunctionOpInterface>(callee);
        if (funcOp) {
          graph[caller].emplace_back(
              std::pair<CallOpInterface, FunctionOpInterface>(callOp, funcOp));
          visited.insert(funcOp);
        }
      }
    });
    // Find roots
    moduleOp.walk([&](FunctionOpInterface funcOp) {
      if (!visited.count(funcOp)) {
        roots.push_back(funcOp);
      }
    });
  }

  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void doWalk(FunctionOpInterface funcOp,
              DenseSet<FunctionOpInterface> &visited, UpdateEdgeFn updateEdgeFn,
              UpdateNodeFn updateNodeFn) {
    if (visited.count(funcOp)) {
      llvm::report_fatal_error("Cycle detected in call graph");
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PreOrder) {
      updateNodeFn(funcOp);
    }
    for (auto [callOp, callee] : graph[funcOp]) {
      if constexpr (UpdateEdgeOrder == WalkOrder::PreOrder) {
        updateEdgeFn(callOp, callee);
      }
      doWalk<UpdateEdgeOrder, UpdateNodeOrder>(callee, visited, updateEdgeFn,
                                               updateNodeFn);
      if constexpr (UpdateEdgeOrder == WalkOrder::PostOrder) {
        updateEdgeFn(callOp, callee);
      }
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PostOrder) {
      updateNodeFn(funcOp);
    }
    visited.erase(funcOp);
  }

protected:
  ModuleOp moduleOp;
  DenseMap<FunctionOpInterface,
           SmallVector<std::pair<CallOpInterface, FunctionOpInterface>>>
      graph;
  FuncDataMapT funcMap;
  SmallVector<FunctionOpInterface> roots;
};
// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

triton::MakeTensorPtrOp getMakeTensorPtrOp(Value v);

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
