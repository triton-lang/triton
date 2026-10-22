#include "triton/Dialect/TritonNvidiaGPU/IR/TMAUtils.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LayoutUtils.h"

namespace mlir::triton::nvidia_gpu {
namespace ttg = gpu;

static LinearLayout getMessageToPackedOffsetLayout(ttg::MemDescType type,
                                                   ttg::TMAMode mode) {
  auto *ctx = type.getContext();
  auto message = StringAttr::get(ctx, "msg");
  auto shapePerCTA = ttg::getShapePerCTA(type);
  auto blockShape = getTMABlockShape(type, /*packedSize=*/true, mode);
  auto dimensions = standardOutDimNames(ctx, shapePerCTA.size());
  LinearLayout layout;
  for (unsigned dim = 0; dim < shapePerCTA.size(); ++dim)
    layout *=
        LinearLayout::strided1D(shapePerCTA[dim] / blockShape[dim],
                                blockShape[dim], message, dimensions[dim]);
  return layout * ttg::getCGALayout(type.getEncoding()).getLinearLayout();
}

TMAMessageLayouts getTMAMessageLayouts(ttg::MemDescType type,
                                       ttg::TMAMode mode) {
  auto *ctx = type.getContext();
  auto block = StringAttr::get(ctx, "block");
  auto packed = getMessageToPackedOffsetLayout(type, mode);
  auto unpacked = packed;
  if (isFp4Padded(type.getEncoding())) {
    auto message = StringAttr::get(ctx, "msg");
    auto lastDim = StringAttr::get(ctx, "dim" + Twine(type.getRank() - 1));
    unpacked = LinearLayout::zeros1D(1, message, lastDim, 2) * packed;
  }
  auto shared =
      invertAndComposeLocal(ttg::toLinearLayout(type), packed, {block});
  return {std::move(packed), std::move(unpacked), std::move(shared)};
}

TMAGatherScatterLayouts
getTMAGatherScatterLayouts(RankedTensorType xOffsetsType,
                           ttg::MemDescType type) {
  auto *ctx = type.getContext();
  auto column = StringAttr::get(ctx, "dim1");
  auto message = StringAttr::get(ctx, "msg");
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto block = StringAttr::get(ctx, "block");
  auto xOffsets =
      ttg::toLinearLayout(xOffsetsType).removeZeroBasesAlongDim(reg);
  assert(xOffsets.getInDimSize(reg) >= 4 &&
         "must have at least 4 x offsets per warp");
  assert(xOffsets.getBasis(reg, 0).front() == 1 &&
         xOffsets.getBasis(reg, 1).front() == 2 &&
         "x offsets are not grouped by 4 contiguous elements");
  auto encoding = cast<ttg::NVMMASharedEncodingAttr>(type.getEncoding());
  auto freeVars = xOffsets.getFreeVariableMasks();
  assert(freeVars.lookup(lane) == xOffsets.getInDimSize(lane) - 1 &&
         "x offsets must be broadcasted across each warp");

  auto shapePerCTA = ttg::getShapePerCTA(type);
  auto blockShape =
      getTMABlockShape(type, /*packedSize=*/true, ttg::TMAMode::Tiled);
  unsigned messageSize = blockShape.back();
  assert(shapePerCTA.back() % messageSize == 0);
  unsigned numMessages = shapePerCTA.back() / messageSize;
  assert(llvm::isPowerOf2_32(numMessages));
  auto messageToColumn =
      LinearLayout::strided1D(numMessages, messageSize, message, column);
  auto messageToOffset =
      getMessageToPackedOffsetLayout(type, ttg::TMAMode::Tiled);
  auto messageLayout = xOffsets * messageToColumn;
  auto bases = messageLayout.getBases();
  for (auto [bit, basis] : llvm::enumerate(bases[block]))
    basis.back() = messageToOffset.getBasis(block, bit, column);
  auto outDims = messageLayout.getOutDims();
  outDims[messageLayout.getOutDimIndex(column)].second = type.getShape().back();
  messageLayout = LinearLayout(std::move(bases), outDims,
                               /*requireSurjective=*/false);

  // gather4 places four row segments consecutively; message starts use the
  // unswizzled allocation layout.
  auto sharedLayout = ttg::nvmmaSharedToLinearLayout(
      ttg::dropPipeliningDim(type.getAllocShape(), type.getEncoding()),
      encoding, ttg::TMAMode::Tiled, /*disableSwizzle=*/true);
  auto messageToShared =
      invertAndComposeLocal(sharedLayout, messageLayout, {block});
  return TMAGatherScatterLayouts{std::move(xOffsets),
                                 std::move(messageToOffset),
                                 std::move(messageToShared),
                                 numMessages,
                                 messageSize,
                                 static_cast<unsigned>(freeVars[warp])};
}

} // namespace mlir::triton::nvidia_gpu
