#include "SubtileFusion.h"
#include "amd/lib/TritonAMDGPUTransforms/Utility.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "triton/Dialect/TritonGPU/IR/CGAEncodingAttr.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir {

namespace {

// Returns `a - b` when the two index values differ by a compile-time constant,
// looking through one `arith.addi` with a constant operand on either side.
// TODO: Decompose indices into a linear form (nested addi/subi, muli/shli by a
// constant) if slice offsets written in other shapes need to be matched.
std::optional<int64_t> getConstantIndexDelta(Value a, Value b) {
  if (a == b)
    return 0;
  if (auto constA = getConstantIntValue(a))
    if (auto constB = getConstantIntValue(b))
      return *constA - *constB;
  auto splitConstantAdd = [](Value v) -> std::pair<Value, int64_t> {
    if (auto add = v.getDefiningOp<arith::AddIOp>()) {
      if (auto c = getConstantIntValue(add.getRhs()))
        return {add.getLhs(), *c};
      if (auto c = getConstantIntValue(add.getLhs()))
        return {add.getRhs(), *c};
    }
    return {v, 0};
  };
  auto [baseA, offsetA] = splitConstantAdd(a);
  auto [baseB, offsetB] = splitConstantAdd(b);
  if (baseA != baseB)
    return std::nullopt;
  return offsetA - offsetB;
}

// Matches `loads`, which share a descriptor, a result type and a block,
// against a grid with exactly one load per cell.
std::optional<SubtileLoadGroup>
matchSubtileLoadGroup(SmallVector<tt::DescriptorLoadOp> loads) {
  llvm::sort(loads, [](tt::DescriptorLoadOp a, tt::DescriptorLoadOp b) {
    return a->isBeforeInBlock(b);
  });
  ArrayRef<int64_t> childShape =
      cast<RankedTensorType>(loads.front().getType()).getShape();
  unsigned rank = childShape.size();

  // Compute the offset of each child from the origin.
  SmallVector<SmallVector<int64_t>> offsets;
  for (tt::DescriptorLoadOp load : loads) {
    SmallVector<int64_t> &offset = offsets.emplace_back();
    for (auto [ref, index] :
         llvm::zip(loads.front().getIndices(), load.getIndices())) {
      std::optional<int64_t> delta = getConstantIndexDelta(index, ref);
      if (!delta)
        return std::nullopt;
      offset.push_back(*delta);
    }
  }

  // Along every dimension the children sit at consecutive multiples of the
  // child size, so they abut without gaps or overlap.
  SmallVector<int64_t> parentShape;
  int64_t numCells = 1;
  for (unsigned d = 0; d < rank; ++d) {
    int64_t low = offsets.front()[d];
    for (ArrayRef<int64_t> offset : offsets)
      low = std::min(low, offset[d]);

    // Normalize the offsets to start at zero.
    SmallVector<int64_t> positions;
    for (SmallVector<int64_t> &offset : offsets)
      positions.push_back(offset[d] -= low);

    // Check that the offsets are consecutive multiples of the child size.
    llvm::sort(positions);
    positions.erase(llvm::unique(positions), positions.end());
    for (auto [i, position] : llvm::enumerate(positions))
      if (position != static_cast<int64_t>(i) * childShape[d])
        return std::nullopt;
    // Tensor shapes are powers of two, so the slice count must be one too.
    if (!llvm::isPowerOf2_64(positions.size()))
      return std::nullopt;
    parentShape.push_back(positions.size() * childShape[d]);
    numCells *= positions.size();
  }

  // One child per cell.
  SmallVector<SmallVector<int64_t>> cells = offsets;
  llvm::sort(cells);
  if (numCells != static_cast<int64_t>(loads.size()) ||
      llvm::unique(cells) != cells.end())
    return std::nullopt;

  auto isOrigin = [](ArrayRef<int64_t> offset) {
    return llvm::all_of(offset, [](int64_t x) { return x == 0; });
  };
  tt::DescriptorLoadOp origin =
      loads[llvm::find_if(offsets, isOrigin) - offsets.begin()];
  return SubtileLoadGroup{std::move(loads), std::move(offsets), origin,
                          std::move(parentShape)};
}

// The children's padded layout resized to the parent tile.
ttg::PaddedSharedEncodingAttr
getWholeTileEncoding(ttg::PaddedSharedEncodingAttr childEnc,
                     ArrayRef<int64_t> parentShape, unsigned elemBitWidth) {
  unsigned parentRow = parentShape[childEnc.getOrder()[0]];
  unsigned maxInterval = 256u * 32u / elemBitWidth;
  SmallVector<std::pair<unsigned, unsigned>> intervalPads;
  for (auto [interval, padding] :
       llvm::zip(childEnc.getIntervals(), childEnc.getPaddings())) {
    unsigned widened = std::max(interval, parentRow);
    intervalPads.emplace_back(widened <= maxInterval ? widened : interval,
                              padding);
  }
  return ttg::PaddedSharedEncodingAttr::get(childEnc.getContext(), intervalPads,
                                            childEnc.getOrder(), parentShape,
                                            ttg::getCGALayout(childEnc));
}

} // namespace

// Returns the pipelined descriptor loads that can share one whole-tile TDM
// copy. A SubtileLoadGroup is created only when all of the following hold:
// - At least two loads share a descriptor, a result type, and a block.
// - Each load's descriptor comes from make_tensor_descriptor, and its shared
//   encoding is a padded layout with a single interval, not split across
//   CTAs.
// - Each load returns a tensor of the descriptor's rank (no rank-reducing
//   loads).
// - The loads use the same shared encoding and the same pipeline distance.
// - Their indices differ by compile-time constants and form a complete grid:
//   along each dimension they sit at 0, childSize, 2 * childSize, ... with
//   one load per cell and no gaps or overlap, and each dimension holds a
//   power-of-two number of them.
// - The indices of the cell at offset zero are available before the first
//   load, so the fused copy can be built there.
SmallVector<SubtileLoadGroup>
findSubtileLoadGroups(const LoadToInfoMap &loadToInfo) {
  llvm::MapVector<std::tuple<Value, Type, Block *>,
                  SmallVector<tt::DescriptorLoadOp>>
      candidates;
  for (auto &[op, info] : loadToInfo) {
    auto load = dyn_cast<tt::DescriptorLoadOp>(op);
    auto sharedEnc =
        dyn_cast_or_null<ttg::PaddedSharedEncodingAttr>(info.sharedEncoding);
    // The parent's encoding is the child's resized to the parent shape, which
    // is defined for padded layouts, and the parent descriptor is a copy of
    // the child's with a larger block, so it must come from a visible
    // make_tensor_descriptor. The indices of a rank-reducing load address
    // the descriptor's dimensions rather than the result's, so they cannot
    // be compared with the result shape.
    if (!load || !sharedEnc ||
        !load.getDesc().getDefiningOp<tt::MakeTensorDescOp>() ||
        load.getDesc().getType().getBlockType().getRank() !=
            cast<RankedTensorType>(load.getType()).getRank())
      continue;
    // The resize widens each padding interval on its own, which can make two
    // of them equal, and keeps the CGA layout, which would split the parent
    // rather than each child across CTAs.
    if (sharedEnc.getIntervals().size() != 1 ||
        llvm::any_of(ttg::getCTASplitNum(sharedEnc),
                     [](unsigned split) { return split != 1; }))
      continue;
    candidates[{load.getDesc(), load.getType(), load->getBlock()}].push_back(
        load);
  }

  SmallVector<SubtileLoadGroup> groups;
  for (auto &[key, loads] : candidates) {
    if (loads.size() < 2)
      continue;
    // The children share one buffer, so they must agree on its layout and on
    // how far ahead it is filled.
    LoadInfo first = loadToInfo.lookup(loads.front());
    if (llvm::any_of(loads, [&](tt::DescriptorLoadOp load) {
          LoadInfo info = loadToInfo.lookup(load);
          return info.sharedEncoding != first.sharedEncoding ||
                 info.distToUse != first.distToUse;
        }))
      continue;

    std::optional<SubtileLoadGroup> group = matchSubtileLoadGroup(loads);
    if (!group)
      continue;
    // The copy is built in front of the first child from the origin's
    // indices, so they must already be available there.
    Operation *firstChild = group->children.front();
    if (llvm::any_of(group->origin.getIndices(), [&](Value index) {
          Operation *def = index.getDefiningOp();
          return def && def->getBlock() == firstChild->getBlock() &&
                 !def->isBeforeInBlock(firstChild);
        }))
      continue;
    groups.push_back(std::move(*group));
  }
  return groups;
}

void createFusedTDMAsyncCopy(const SubtileLoadGroup &group,
                             ttg::SharedEncodingTrait childSharedEnc,
                             scf::ForOp forOp, Value extractIdx, int numBuffers,
                             LoadToStreamOpMap &loadToStreamOp) {
  tt::DescriptorLoadOp origin = group.origin;
  auto childTy = cast<RankedTensorType>(origin.getType());
  auto parentEnc =
      getWholeTileEncoding(cast<ttg::PaddedSharedEncodingAttr>(childSharedEnc),
                           group.parentShape, childTy.getElementTypeBitWidth());

  auto parentTy = RankedTensorType::get(
      group.parentShape, childTy.getElementType(), childTy.getEncoding());
  Value alloc = triton::createAlloc(forOp, parentTy, origin.getLoc(), parentEnc,
                                    numBuffers);

  auto makeDesc = origin.getDesc().getDefiningOp<tt::MakeTensorDescOp>();
  OpBuilder descBuilder(makeDesc);
  descBuilder.setInsertionPointAfter(makeDesc);
  auto parentDesc = cast<tt::MakeTensorDescOp>(descBuilder.clone(*makeDesc));
  auto childDescTy = cast<tt::TensorDescType>(makeDesc.getType());
  parentDesc.getResult().setType(tt::TensorDescType::get(
      group.parentShape, childDescTy.getElementType(), parentEnc));

  OpBuilder builder(group.children.front());
  Location loc = origin.getLoc();
  Value pred = arith::ConstantIntOp::create(builder, loc, 1, 32);
  auto view = triton::createSingleBufferView(builder, alloc, extractIdx)
                  .getDefiningOp<ttg::MemDescIndexOp>();
  Value desc = createUpdateTDMDescriptorOp(builder, loc, parentDesc.getResult(),
                                           origin.getIndices(), pred);
  auto copy = triton::amdgpu::AsyncTDMCopyGlobalToLocalOp::create(builder, loc,
                                                                  desc, view);
  auto wait =
      triton::amdgpu::AsyncTDMWait::create(builder, loc, copy->getResult(0), 0);

  ttg::MemDescType viewTy = view.getType();
  auto sliceTy =
      ttg::MemDescType::get(childTy.getShape(), childTy.getElementType(),
                            parentEnc, viewTy.getMemorySpace(),
                            viewTy.getMutableMemory(), viewTy.getAllocShape());
  for (auto [constChild, offset] : llvm::zip(group.children, group.offsets)) {
    tt::DescriptorLoadOp child = constChild;
    OpBuilder childBuilder(child);
    SmallVector<int32_t> offsets32(offset.begin(), offset.end());
    auto slice = ttg::MemDescSubsliceOp::create(childBuilder, child.getLoc(),
                                                sliceTy, view, offsets32);
    auto localLoad = tt::replaceUsesWithLocalLoad(
        childBuilder, child->getResult(0), slice, wait);
    loadToStreamOp[child] = TDMChainOps{copy, wait, localLoad};
  }
}

} // namespace mlir
