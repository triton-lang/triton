#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>
#include <string>

namespace mlir {

class ReduceOpHelper {
public:
  explicit ReduceOpHelper(triton::ReduceOp op) : op(op) {
    srcTy = op.operand().getType().cast<RankedTensorType>();
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

private:
  triton::ReduceOp op;
  RankedTensorType srcTy{};
};

bool isSharedEncoding(Value value);

bool maybeSharedAllocationOp(Operation *op);

bool maybeAliasOp(Operation *op);

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

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
