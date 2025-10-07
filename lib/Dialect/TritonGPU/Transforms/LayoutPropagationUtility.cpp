#include "triton/Dialect/TritonGPU/Transforms/LayoutPropagationUtility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include <optional>
#include <utility>

namespace mlir::triton::gpu {

std::optional<std::pair<triton::LoadOp, LinearLayout>>
inferSourceLoadLayout(const LinearLayout &dstLayout, Operation *defOp) {
  if (!defOp)
    return std::nullopt;
  return inferSourceLoadLayout(
      LinearEncodingAttr::get(defOp->getContext(), dstLayout), defOp);
}

std::optional<std::pair<triton::LoadOp, LinearLayout>>
inferSourceLoadLayout(LinearEncodingAttr dstLayout, Operation *defOp) {
  Attribute curLayout = dstLayout;
  Operation *curOp = defOp;
  while (curOp) {
    if (isa<triton::LoadOp>(curOp))
      break; // Found the load op; we are done here.

    if (auto cvtOp = dyn_cast<ConvertLayoutOp>(curOp)) {
      // For convert op we keep the current layout to push through further.
      curOp = cvtOp.getSrc().getDefiningOp();
    } else {
      if (curOp->getNumOperands() != 1)
        break;
      curLayout = inferSrcEncoding(curOp, curLayout);
      curOp = curOp->getOperand(0).getDefiningOp();
    }
  }
  auto loadOp = dyn_cast_or_null<triton::LoadOp>(curOp);
  if (!loadOp)
    return std::nullopt;
  auto loadType = dyn_cast<RankedTensorType>(loadOp.getType());
  if (!loadType)
    return std::nullopt;

  return std::make_pair(
      loadOp,
      toLinearLayout(loadType.getShape(), cast<LinearEncodingAttr>(curLayout)));
}

} // namespace mlir::triton::gpu
