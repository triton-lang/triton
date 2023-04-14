#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>
#include <string>

namespace mlir {

class ReduceOpHelper {
public:
  explicit ReduceOpHelper(triton::ReduceOp op) : op(op) {
    srcTy = op.getOperand().getType().cast<RankedTensorType>();
  }

  ArrayRef<int64_t> getSrcShape() { return srcTy.getShape(); }

  Attribute getSrcLayout() { return srcTy.getEncoding(); }

  bool isFastReduction();

  unsigned getInterWarpSize();

  unsigned getIntraWarpSize();

  unsigned getThreadsReductionAxis();

  SmallVector<unsigned> getScratchConfigBasic();

  SmallVector<SmallVector<unsigned>> getScratchConfigsFast();

  unsigned getScratchSizeInBytes();

  bool isSupportedLayout();

private:
  triton::ReduceOp op;
  RankedTensorType srcTy{};
};

bool isSharedEncoding(Value value);

bool maybeSharedAllocationOp(Operation *op);

bool maybeAliasOp(Operation *op);

bool supportMMA(triton::DotOp op, int version);

bool supportMMA(Value value, int version);

Type getElementType(Value value);

std::string getValueOperandName(Value value, AsmState &state);

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

// output[i] = input[order[i]]
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

template <typename T> T highestPowOf2Divisor(T n) {
  if (n == 0) {
    return (static_cast<T>(1) << (sizeof(T) * 8 - 2));
  }
  return (n & (~(n - 1)));
}

bool isSingleValue(Value value);

bool isMmaToDotShortcut(RankedTensorType &srcTy, RankedTensorType &dstTy);

/// Multi-root DAG topological sort.
/// Performs a topological sort of the Operation in the `toSort` SetVector.
/// Returns a topologically sorted SetVector.
/// It is faster than mlir::topologicalSort because it prunes nodes that have
/// been visited before.
SetVector<Operation *>
multiRootTopologicalSort(const SetVector<Operation *> &toSort);

// This uses the toplogicalSort above
SetVector<Operation *>
multiRootGetSlice(Operation *op, TransitiveFilter backwardFilter = nullptr,
                  TransitiveFilter forwardFilter = nullptr);

// Create a basic DataFlowSolver with constant and dead code analysis included.
std::unique_ptr<DataFlowSolver> createDataFlowSolver();

template <typename T> class CallGraph {
public:
  using FuncDataMapT = DenseMap<triton::FuncOp, T>;
  CallGraph(ModuleOp moduleOp) : moduleOp(moduleOp) { build(); }

  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void apply(UpdateEdgeFn updateEdgeFn, UpdateNodeFn updateNodeFn) {
    DenseSet<triton::CallOp> visited;
    for (auto root : roots) {
      doApply<UpdateEdgeOrder, UpdateNodeOrder>(root, visited, updateEdgeFn,
                                                updateNodeFn);
    }
  }

  T *getFuncData(triton::FuncOp funcOp) {
    if (funcMap.count(funcOp)) {
      return &funcMap[funcOp];
    }
    return nullptr;
  }

  ModuleOp getModuleOp() const { return moduleOp; }

  SmallVector<triton::FuncOp> getRoots() const { return roots; }

private:
  void build() {
    SymbolTableCollection symbolTable;
    DenseMap<Operation *, Operation *> parentMap;
    moduleOp.walk([&](Operation *op) {
      auto parent = op->getParentOfType<triton::FuncOp>();
      if (auto callOpInterface = dyn_cast<CallOpInterface>(op)) {
        auto callOp = cast<triton::CallOp>(op);
        auto *callee = callOpInterface.resolveCallable(&symbolTable);
        auto funcOp = dyn_cast_or_null<triton::FuncOp>(callee);
        if (funcOp)
          callGraph[parent].emplace_back(
              std::pair<triton::CallOp, triton::FuncOp>(callOp, funcOp));
      }
      parentMap[op] = parent;
      if (parent == op)
        roots.push_back(parent);
    });
  }

  template <WalkOrder UpdateEdgeOrder = WalkOrder::PreOrder,
            WalkOrder UpdateNodeOrder = WalkOrder::PreOrder,
            typename UpdateEdgeFn, typename UpdateNodeFn>
  void doApply(triton::FuncOp funcOp, DenseSet<triton::CallOp> &visited,
               UpdateEdgeFn updateEdgeFn, UpdateNodeFn updateNodeFn) {
    if constexpr (UpdateNodeOrder == WalkOrder::PreOrder) {
      updateNodeFn(funcOp, funcMap);
    }
    for (auto [callOp, callee] : callGraph[funcOp]) {
      if (visited.count(callOp)) {
        llvm::report_fatal_error("Cycle detected in call graph");
      }
      visited.insert(callOp);
      if constexpr (UpdateEdgeOrder == WalkOrder::PreOrder) {
        updateEdgeFn(callOp, callee);
      }
      updateEdgeFn(callOp, callee);
      if constexpr (UpdateEdgeOrder == WalkOrder::PostOrder) {
        updateEdgeFn(callOp, callee);
      }
      doApply(callee, visited, updateEdgeFn, updateNodeFn);
      visited.erase(callOp);
    }
    if constexpr (UpdateNodeOrder == WalkOrder::PostOrder) {
      updateNodeFn(funcOp, funcMap);
    }
  }

  ModuleOp moduleOp;
  DenseMap<triton::FuncOp,
           SmallVector<std::pair<triton::CallOp, triton::FuncOp>>>
      callGraph;
  FuncDataMapT funcMap;
  SmallVector<triton::FuncOp> roots;
};

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
