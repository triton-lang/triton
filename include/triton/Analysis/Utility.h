#ifndef TRITON_ANALYSIS_UTILITY_H
#define TRITON_ANALYSIS_UTILITY_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <algorithm>
#include <numeric>
#include <string>

namespace mlir {

bool isSharedEncoding(Value value);

bool maybeSharedAllocationOp(Operation *op);

std::string getValueOperandName(Value value, AsmState &state);

template <typename Int> Int product(llvm::ArrayRef<Int> arr) {
  return std::accumulate(arr.begin(), arr.end(), 1, std::multiplies{});
}

template <typename Int> Int ceil(Int m, Int n) { return (m + n - 1) / n; }

} // namespace mlir

#endif // TRITON_ANALYSIS_UTILITY_H
