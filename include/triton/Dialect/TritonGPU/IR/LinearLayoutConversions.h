// Conversions from TritonGPU layouts (e.g. BlockedEncodingAttr) to
// LinearLayout.

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
//   "offset": the n'th element in the allocation, within a particular block
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
LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout);

// Returns true iff the given layout can be converted to a LinearLayout.  If
// this returns false, calling toLinearLayout(layout) will assert.
//
// TODO(jlebar): Remove this once all legacy layouts are supported.
bool toLinearLayoutIsSupported(Attribute layout);

// TODO(jlebar): Helpers to convert a flattened shared layout to vector loads of
// size n, in banks of size k, etc.

} // namespace mlir::triton::gpu
