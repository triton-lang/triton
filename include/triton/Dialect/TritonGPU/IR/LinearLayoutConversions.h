// Conversions from TritonGPU layouts (e.g. BlockedEncodingAttr) to
// LinearLayout.

#ifndef TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
#define TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H

#include <optional>

#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::gpu {

// - BlockedEncodingAttrs have the following input dimensions.
//
//   "register": elements in one thread
//   "lane": threads in a warp
//   "warp": warps in a block/CTA
//   "block": blocks in a cluster
//
// - An n-dimensional SharedEncodingAttr has the following input dimensions.
//
//   "offset": the n'th element in the allocation, within a particular thread
//      block (i.e. within a CTA).  The offset is measured in elements, not
//      bytes.
//   "block": blocks in a cluster
//
// All layouts have the following output dimensions.
//
//  "dimi" for i in 0..n-1: the location in the n'th logical dimension of the
//  output tensor.  These also are not reordered according to the layout's
//  `order`.
//
// You can flatten the input or output dimensions into a single dimension using
// LinearLayout::flattenIns/Outs().
//
// elemBitWidth is the bit width of one element in the layout.  This is required
// to compute the linear layout for MMAv3 (i.e. Hopper) shared layouts (i.e.
// shared layouts with hasLeadingOffset == true) but is otherwise unused.
//
// Returns std::nullopt if the given layout can't be converted to an LL.
// TODO(jlebar): Remove the std::optional once all layouts are supported.
//
std::optional<LinearLayout>
toLinearLayout(ArrayRef<int64_t> shape, Attribute layout,
               std::optional<int32_t> elemBitWidth = std::nullopt);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_IR_LINEARLAYOUTCONVERSIONS_H
