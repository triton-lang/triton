#ifndef TRITON_CPU_ANALYSIS_TENSORPTRSHAPEINFO_H
#define TRITON_CPU_ANALYSIS_TENSORPTRSHAPEINFO_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <optional>
#include <type_traits>

namespace mlir::triton::cpu {

// Lattice value to hold a shape and strides for a tensor pointer.
// If multiple size or stride values are possible for some dimension
// then ShapedType::kDynamic is used for that dimension.
class TensorPtrShapeInfo {
public:
  TensorPtrShapeInfo() = default;

  TensorPtrShapeInfo(ArrayRef<int64_t> shape, ArrayRef<int64_t> strides)
      : shape(shape), strides(strides) {
    assert(shape.size() == strides.size());
  }

  ArrayRef<int64_t> getShape() const { return shape; }
  ArrayRef<int64_t> getStrides() const { return strides; }

  int64_t getRank() const { return static_cast<int64_t>(shape.size()); }
  int64_t getSize(int64_t dim) const { return shape[dim]; }
  int64_t getStride(int64_t dim) const { return strides[dim]; }

  bool operator==(const TensorPtrShapeInfo &other) const {
    return shape == other.shape && strides == other.strides;
  }

  static TensorPtrShapeInfo join(const TensorPtrShapeInfo &lhs,
                                 const TensorPtrShapeInfo &rhs);

  static TensorPtrShapeInfo getPessimisticValueState(Value value);

  void print(raw_ostream &os) const {
    os << "shape = [";
    llvm::interleaveComma(shape, os);
    os << "], strides = [";
    llvm::interleaveComma(strides, os);
    os << "]";
  }

private:
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
};

using TensorPtrShapeInfoMapT = DenseMap<Value, TensorPtrShapeInfo>;
class ModuleTensorPtrShapeInfoAnalysis
    : public CallGraph<TensorPtrShapeInfoMapT> {
public:
  explicit ModuleTensorPtrShapeInfoAnalysis(ModuleOp moduleOp)
      : CallGraph<TensorPtrShapeInfoMapT>(moduleOp) {
    SmallVector<FunctionOpInterface> funcs;
    for (auto root : getRoots()) {
      walk<WalkOrder::PreOrder, WalkOrder::PostOrder>(
          // Pre-order edge walk callback
          [](CallOpInterface callOp, FunctionOpInterface funcOp) {},
          // Post-order node walk callback
          [&](FunctionOpInterface funcOp) {
            funcs.push_back(funcOp);
            funcMap.try_emplace(funcOp, TensorPtrShapeInfoMapT{});
          });
    }
    SetVector<FunctionOpInterface> sortedFuncs(funcs.begin(), funcs.end());
    SymbolTableCollection symbolTable;
    for (auto funcOp : llvm::reverse(sortedFuncs)) {
      initialize(funcOp);
      funcOp.walk([&](CallOpInterface callOp) {
        auto callee =
            dyn_cast<FunctionOpInterface>(callOp.resolveCallable(&symbolTable));
        update(callOp, callee);
      });
    }
  }

  TensorPtrShapeInfo *getPtrShapeInfo(Value value) {
    auto funcOp =
        value.getParentRegion()->getParentOfType<FunctionOpInterface>();
    auto *axisInfoMap = getFuncData(funcOp);
    if (!axisInfoMap) {
      return nullptr;
    }
    auto it = axisInfoMap->find(value);
    if (it == axisInfoMap->end()) {
      return nullptr;
    }
    return &(it->second);
  }

private:
  void initialize(FunctionOpInterface funcOp);
  void update(CallOpInterface callOp, FunctionOpInterface funcOp);
};

} // namespace mlir::triton::cpu

#endif // TRITON_CPU_ANALYSIS_TENSORPTRSHAPEINFO_H
