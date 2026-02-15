#ifndef TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_PROPAGATION_UTILITY_H_
#define TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_PROPAGATION_UTILITY_H_

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Tools/LinearLayout.h"
#include <optional>

namespace mlir::triton::gpu {

// Given the result |dstLayout|, infer the source layout that we should use for
// global load if we propagate through op def chain of |defOp|. Returns
// std::nullopt if fails to infer or cannot reach a global load.
std::optional<std::pair<triton::LoadOp, LinearLayout>>
inferSourceLoadLayout(const LinearLayout &dstLayout, Operation *defOp);
std::optional<std::pair<triton::LoadOp, LinearLayout>>
inferSourceLoadLayout(LinearEncodingAttr dstLayout, Operation *defOp);

} // namespace mlir::triton::gpu

#endif // TRITON_DIALECT_TRITONGPU_TRANSFORMS_LAYOUT_PROPAGATION_UTILITY_H_
