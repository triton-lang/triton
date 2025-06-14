#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_UTILS_UTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_UTILS_UTILITY_H_

#include "llvm/ADT/ArrayRef.h"
#include <cassert>
#include <vector>
namespace mlir::LLVM::AMD {

template <typename T, typename U, typename BinaryOp>
std::vector<unsigned> multiDimElementwise(const ArrayRef<T> &lhs,
                                          const ArrayRef<U> &rhs, BinaryOp op) {
  assert(lhs.size() == rhs.size() && "Input dimensions must match");
  std::vector<unsigned> result;
  result.reserve(lhs.size());
  for (size_t i = 0, n = lhs.size(); i < n; ++i) {
    unsigned a = static_cast<unsigned>(lhs[i]);
    unsigned b = static_cast<unsigned>(rhs[i]);
    result.push_back(op(a, b));
  }
  return result;
}
} // namespace mlir::LLVM::AMD
#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_UTILS_UTILITY_H_
