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
  // Return the number of elements of the scratch space needed for scan
  // lowering.
  unsigned getScratchSizeInElems();

  // Stride between contiguous element along axis dim.
  unsigned getAxisElementStride();
  // Stride between contiguous threads along axis dim.
  unsigned getAxisThreadStride();
  // Stride between contiguous blocks along axis dim.
  unsigned getAxisBlockStride();

  Location getLoc() { return scanOp.getLoc(); }
  unsigned getAxis() { return scanOp.getAxis(); }
  triton::gpu::BlockedEncodingAttr getEncoding();
  llvm::ArrayRef<int64_t> getShape() { return srcShape; }
  unsigned getNumOperands() { return scanOp.getNumOperands(); }
  SmallVector<Type> getElementTypes() { return srcElementTypes; }
  Attribute getSrcLayout() { return srcEncoding; }
  Region &getCombineOp();

private:
  triton::ScanOp scanOp;
  Attribute srcEncoding;
  llvm::ArrayRef<int64_t> srcShape;
  SmallVector<Type> srcElementTypes;
};

// Decomposes a reshape into simpler pieces.
//
// As an example, suppose we have a reshape from [4,4,4] to [2,2,8,2].
// You might explain what this does as follows.
//
//  - Split the first input dimension into [2,2].
//  - Take the remaining two input dimensions, merge them into a single [16]
//    dim, and then split that into [8,2].
//
// In general, a reshape can be described a sequence of smushing one or more
// input dimensions together and then breaking them apart into one or more
// output dimensions.  So we could represent the example above as follows.
//
//   [
//     ([0], [0, 1]),  # input dim [0] -> output dims [0, 1]
//     ([1, 2], [2, 3]),  # input dims [1, 2] -> output dims [2, 3]
//   ]
//
// Notice that the input dims (first tuple elems) appear in sequential order if
// you read left-to-right-top-to-bottom, and so do the output dims.
//
// This function returns the above decomposition.
SmallVector<std::pair<SmallVector<int64_t>, SmallVector<int64_t>>>
getReshapeDecomposition(ArrayRef<int64_t> srcShape, ArrayRef<int64_t> dstShape);

bool maybeSharedAllocationOp(Operation *op);

bool maybeAliasOp(Operation *op);

bool supportMFMA(triton::DotOp op);

bool supportMMA(triton::DotOp op, int version);

bool supportMMA(Value value, int version);

bool isSingleValue(Value value);

bool isMfmaToDotShortcut(RankedTensorType &srcTy, RankedTensorType &dstTy);

bool isMmaToDotShortcut(RankedTensorType srcTy, RankedTensorType dstTy);

bool isMmaToMmaShortcut(RankedTensorType srcTy, RankedTensorType dstTy);

// Return true if the src and dst layout match.
bool matchMmaV3AndDotOperandLayout(RankedTensorType srcTy,
                                   RankedTensorType dstTy);

// TODO: Move utility functions that belong to ConvertLayoutOp to class
// ConvertLayoutOpHelper in the future
bool shouldUseDistSmem(Attribute srcLayout, Attribute dstLayout);

template <typename T, typename U> SmallVector<T> convertType(ArrayRef<U> in) {
  SmallVector<T> out;
  for (const auto &i : in)
    out.push_back(T(i));
  return out;
}
template <typename T, typename VecU>
SmallVector<T> convertType(const VecU &in) {
  return convertType<T>(ArrayRef(in));
}

template <typename Int> Int product(llvm::ArrayRef<Int> arr) {
  return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies{});
}
template <typename VecT> auto product(const VecT &vec) {
  return product(llvm::ArrayRef(vec));
}

// TODO(jlebar): Rename to ceilOfRatio.
template <typename Int> Int ceil(Int m, Int n) { return (m + n - 1) / n; }

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

namespace triton {

// Many functions here have two overloads, fn(ArrayRef<T>) and fn(const VecT&).
// This is helpful because C++ won't both convert a vector to ArrayRef *and*
// infer the proper type T in one step.  So without the second overload, we
// would have to explicitly convert most arguments to ArrayRef at the callsite.

// Better version of llvm::join.  This one works when T is an integer or any
// other type which defines operator<<(raw_ostream).
template <typename T> std::string join(ArrayRef<T> elems, StringRef sep) {
  std::string ret;
  llvm::raw_string_ostream s(ret);
  if (elems.empty()) {
    return ret;
  }

  s << elems[0];
  for (int i = 1; i < elems.size(); i++) {
    s << sep << elems[i];
  }
  return ret;
}

template <typename VecT> std::string join(const VecT &elems, StringRef sep) {
  return join(ArrayRef(elems), sep);
}

template <typename T, typename U>
SmallVector<T> applyPermutation(ArrayRef<T> vec, ArrayRef<U> permutation) {
  static_assert(std::is_integral_v<U>);
  assert(vec.size() == permutation.size());

  // Check that `permutation` is actually a permutation.
#ifndef NDEBUG
  SmallVector<U> sortedPerm(permutation);
  llvm::sort(sortedPerm);
  for (U i = 0; i < static_cast<U>(sortedPerm.size()); i++) {
    assert(sortedPerm[i] == i);
  }
#endif

  SmallVector<T> ret;
  ret.reserve(vec.size());
  for (const U &i : permutation) {
    ret.push_back(vec[i]);
  }
  return ret;
}

template <typename VecT, typename PermT>
auto applyPermutation(const VecT &vec, const PermT &permutation) {
  return applyPermutation(ArrayRef(vec), ArrayRef(permutation));
}

template <typename T>
[[nodiscard]] SmallVector<T> inversePermutation(ArrayRef<T> permutation) {
  // Check that `permutation` is actually a permutation.
#ifndef NDEBUG
  SmallVector<T> sortedPerm(permutation);
  llvm::sort(sortedPerm);
  for (int i = 0; i < sortedPerm.size(); ++i) {
    assert(sortedPerm[i] == i);
  }
#endif

  SmallVector<T> ret(permutation.size());
  for (int i = 0; i < permutation.size(); ++i) {
    ret[permutation[i]] = i;
  }
  return ret;
}

template <typename VecT>
[[nodiscard]] auto inversePermutation(const VecT &permutation) {
  return inversePermutation(ArrayRef(permutation));
}

template <typename T, typename U>
[[nodiscard]] SmallVector<T> gather(ArrayRef<T> elems, ArrayRef<U> indices) {
  SmallVector<T> ret;
  ret.reserve(indices.size());
  for (const U &i : indices) {
    ret.push_back(elems[i]);
  }
  return ret;
}

template <typename VecT, typename IdxT>
[[nodiscard]] auto gather(const VecT &elems, const IdxT &indices) {
  return gather(ArrayRef(elems), ArrayRef(indices));
}

// Is `vec` [0, 1, ..., n]?  Returns true on empty list.
template <typename T> bool isIota(ArrayRef<T> vec) {
  static_assert(std::is_integral_v<T>);
  for (T i = 0; i < vec.size(); ++i) {
    if (vec[i] != i) {
      return false;
    }
  }
  return true;
}

template <typename VecT> bool isIota(const VecT &vec) {
  return isIota(ArrayRef(vec));
}

// Is `vals` some permutation of the numbers 0..(vals.size()-1)?
template <typename T> bool isPermutationOfIota(ArrayRef<T> vals) {
  SmallVector<T> sorted(vals);
  llvm::sort(sorted);
  return isIota(sorted);
}

template <typename VecT> bool IsPermutationOfIota(const VecT &vec) {
  return isPermutationOfIota(ArrayRef(vec));
}

// Is `vec` [i, i+1, ..., i+n]?  Returns true on empty list.
template <typename T> bool isConsecutive(ArrayRef<T> vec) {
  static_assert(std::is_integral_v<T>);
  for (int i = 1; i < vec.size(); i++) {
    if (vec[i] != vec[i - 1] + 1) {
      return false;
    }
  }
  return true;
}

template <typename VecT> bool isConsecutive(const VecT &vec) {
  return isConsecutive(ArrayRef(vec));
}

} // namespace triton
} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
