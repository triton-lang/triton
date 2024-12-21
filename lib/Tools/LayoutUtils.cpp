#include "triton/Tools/LayoutUtils.h"

namespace mlir::triton {

static bool checkSquareSublayout(const LinearLayout &ll,
                                 ArrayRef<StringAttr> dimNames,
                                 function_ref<bool(int, int32_t)> checkBasis) {
  // The empty layout is the identity
  if (dimNames.size() == 0) {
    return true;
  }
  // Check that the input-output sizes are the same
  LinearLayout sl = ll.sublayout(dimNames, dimNames);
  for (StringAttr dim : dimNames) {
    if (ll.getInDimSize(dim) != ll.getOutDimSize(dim)) {
      return false;
    }
  }
  // Once the inputs and output dimensions are the same, we can just check
  // that the basis for the single remaining dimension is the identity.
  sl = sl.flattenIns().flattenOuts();
  const auto &inDimBases = sl.getBases().begin()->second;
  for (auto [b, basis] : llvm::enumerate(inDimBases)) {
    if (!checkBasis(b, basis[0])) {
      return false;
    }
  }
  return true;
}

bool squareSublayoutIsIdentity(const LinearLayout &ll,
                               ArrayRef<StringAttr> dimNames) {
  return checkSquareSublayout(
      ll, dimNames, [](int b, int32_t basis) { return basis == (1 << b); });
}

bool squareSublayoutIsPermutation(const LinearLayout &ll,
                                  ArrayRef<StringAttr> dimNames) {
  int32_t mask = 0;
  return checkSquareSublayout(ll, dimNames, [&](int b, int32_t basis) {
    if (!llvm::isPowerOf2_32(basis))
      return false;
    if (mask & basis)
      return false; // check if this bit is already set
    mask |= basis;
    return true;
  });
}

} // namespace mlir::triton
