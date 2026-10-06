#ifndef TRITON_ANALYSIS_AFFINEADDRESS_H
#define TRITON_ANALYSIS_AFFINEADDRESS_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <optional>
#include <utility>

namespace mlir::triton {

/// Recognize identical SSA values or duplicated pure, regionless expressions
/// before CSE. The work budget bounds recursive structural comparison.
bool haveEquivalentPureExpressions(Value lhs, Value rhs, unsigned budget = 64);

/// Scalar pointer base plus start + sum(index[d] * steps[d]). An empty base
/// denotes an integer tensor. Every intermediate integer expression must be
/// representable in its original signed width; wraparound is not reassociated.
struct AffineAddress {
  Value base;
  int64_t start = 0;
  SmallVector<int64_t> steps;
  std::optional<std::pair<int64_t, int64_t>>
  bounds(ArrayRef<int64_t> shape) const;
  std::optional<int64_t> linearStep(ArrayRef<int64_t> shape) const;
};

/// A logical contiguous access, independent of load operations and encodings.
/// No alignment, mask, memory-dependence or target profitability claim is made.
struct ContiguousAccessPlan {
  AffineAddress address;
  SmallVector<int64_t> shape;
  unsigned indexBitWidth;
};

/// Bounded, memoized analysis of constant-affine offsets from a scalar base.
/// Construct per query/rewrite; cached entries must not survive IR mutation.
class AffineAddressAnalysis {
public:
  std::optional<AffineAddress> get(Value value, unsigned depth = 0);
  /// Returns rhs - lhs when it is the same constant at every coordinate.
  std::optional<int64_t> getConstantOffsetDifference(Value lhs, Value rhs);
  /// Interleave a power-of-two number of pointer tensors in the given order.
  /// Succeeds only when adjacent components fill a contiguous innermost range.
  std::optional<ContiguousAccessPlan>
  planInterleavedAccess(ArrayRef<Value> ptrs);

private:
  std::optional<AffineAddress> compute(Value value, unsigned depth);
  DenseMap<Value, std::optional<AffineAddress>> cache;
};

} // namespace mlir::triton
#endif
