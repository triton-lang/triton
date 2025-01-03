#ifndef TRITON_TOOLS_LAYOUTUTILS_H
#define TRITON_TOOLS_LAYOUTUTILS_H

#include "triton/Tools/LinearLayout.h"

namespace mlir::triton {
// Is the sublayout defined from dimNames to dimNames the identity?
// In particular, is the input and  output size in these dimensions
// the same, and are the bases the identity?
bool squareSublayoutIsIdentity(const LinearLayout &ll,
                               ArrayRef<StringAttr> dimNames);
} // namespace mlir::triton

#endif // TRITON_TOOLS_LAYOUTUTILS_H
