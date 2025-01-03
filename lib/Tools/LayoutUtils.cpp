#include "triton/Tools/LayoutUtils.h"

namespace mlir::triton {

bool squareSublayoutIsIdentity(const LinearLayout &ll,
                               ArrayRef<StringAttr> dimNames) {
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
  int b = 0;
  const auto &inDimBases = sl.getBases().begin()->second;
  for (auto basis : inDimBases) {
    if (basis[0] != (1 << b)) {
      return false;
    }
    b++;
  }
  return true;
}

} // namespace mlir::triton
