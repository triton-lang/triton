#include "triton/Dialect/TritonGPU/Transforms/DescriptorMemoryLayouts.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/PriorityWorklist.h"
#include <unordered_set>

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir::triton::gpu {

static SmallVector<int64_t> expandToRank(ArrayRef<int64_t> shape, int rank) {
  SmallVector<int64_t> result(rank, 1);
  assert(shape.size() <= rank);
  auto rankDiff = rank - shape.size();
  std::copy(shape.begin(), shape.end(), result.begin() + rankDiff);
  return result;
}

CGAEncodingAttr updateCGALayoutForShape(CGAEncodingAttr cgaLayout,
                                        ArrayRef<int64_t> shape) {
  auto rank = shape.size();
  if (cgaLayout.getRank() == rank)
    return cgaLayout;

  auto ctx = cgaLayout.getContext();
  if (cgaLayout.getRank() > rank) {
    auto ll = cgaLayout.getLinearLayout();
    // Broadcast over the first rankDiff dims
    unsigned rankDiff = cgaLayout.getRank() - rank;
    for (int i = 0; i < rankDiff; ++i) {
      ll = removeStandardDim(ll, 0);
    }
    return CGAEncodingAttr::get(ctx, std::move(ll));
  }
  // For rank-reducing loads, we need to rank-increase the CTA Layout
  auto rankDiff = rank - cgaLayout.getRank();
  for (unsigned i = 0; i < rankDiff; ++i) {
    assert(shape[i] == 1 && "Should only happen for rank-reducing loads");
  }
  auto ll = cgaLayout.getLinearLayout();
  auto kBlock = *ll.getInDimNames().begin();
  auto standardOuts = standardOutDimNames(ctx, rank);
  // Append to front
  for (int i = cgaLayout.getRank(); i < rank; ++i) {
    ll = LinearLayout::identity1D(1, kBlock, standardOuts[i]) * ll;
  }
  // Rename out dims to dim0..dimn-1
  auto dimSizes = ll.getOutDims();
  for (auto [i, dim] : llvm::enumerate(standardOuts)) {
    dimSizes[i].first = dim;
  }
  ll = LinearLayout(ll.getBases(), dimSizes, false);
  return CGAEncodingAttr::get(ctx, std::move(ll));
}

SharedEncodingTrait updateEncodingForShape(Operation *op,
                                           SharedEncodingTrait encoding,
                                           RankedTensorType tensorType) {
  auto ctx = encoding.getContext();
  auto cgaLayout = getCGALayout(encoding);
  if (auto nvmmaEnc = dyn_cast<NVMMASharedEncodingAttr>(encoding)) {
    auto existingCga = nvmmaEnc.getCGALayout();
    if (!existingCga)
      return nvmmaEnc;

    auto newCgaEnc = updateCGALayoutForShape(cgaLayout, tensorType.getShape());
    return NVMMASharedEncodingAttr::get(
        ctx, nvmmaEnc.getSwizzlingByteWidth(), nvmmaEnc.getTransposed(),
        nvmmaEnc.getElementBitWidth(), nvmmaEnc.getFp4Padded(), newCgaEnc);
  }
  if (auto swizEnc = dyn_cast<SwizzledSharedEncodingAttr>(encoding)) {
    auto existingCga = swizEnc.getCGALayout();
    if (!existingCga)
      return swizEnc;

    auto rank = tensorType.getRank();
    auto oldOrder = swizEnc.getOrder();
    SmallVector<unsigned> order;
    for (int i = 0; i + oldOrder.size() < rank; ++i)
      order.push_back(rank - i - 1);
    for (int i = 0; i < oldOrder.size(); ++i) {
      // If it is a rank-reducing load, we need to drop the last dimensions.
      if (oldOrder[i] >= rank)
        continue;
      order.push_back(oldOrder[i]);
    }
    auto newCgaEnc = updateCGALayoutForShape(cgaLayout, tensorType.getShape());
    return SwizzledSharedEncodingAttr::get(
        ctx, swizEnc.getVec(), swizEnc.getPerPhase(), swizEnc.getMaxPhase(),
        order, newCgaEnc);
  }
  if (auto paddedEnc = dyn_cast<ttg::PaddedSharedEncodingAttr>(encoding)) {
    auto existingCga = paddedEnc.getCGALayout();
    if (!existingCga)
      return paddedEnc;

    auto newCgaEnc =
        ttg::updateCGALayoutForShape(cgaLayout, tensorType.getShape());
    auto rank = tensorType.getRank();
    SmallVector<unsigned> order(rank);
    std::iota(order.rbegin(), order.rend(), 0);
    auto shape = tensorType.getShape();
    SmallVector<std::pair<unsigned, unsigned>> intervalPads;
    for (auto [interval, padding] :
         llvm::zip(paddedEnc.getIntervals(), paddedEnc.getPaddings()))
      intervalPads.push_back({interval, padding});
    return ttg::PaddedSharedEncodingAttr::get(ctx, intervalPads, order, shape,
                                              newCgaEnc);
  }

  constexpr auto msg = "Internal Error: Unhandled tensor descriptor encoding";
  if (op)
    op->emitError() << msg;
  llvm::report_fatal_error(msg);
}

// Build shared encoding for a tensor descriptor by applying callback to adjust
// for block shape of the descriptor
TensorDescType getTensorDescTypeWithEncoding(Operation *op,
                                             RankedTensorType existingTy,
                                             Attribute encoding) {
  auto sharedEnc = cast<SharedEncodingTrait>(encoding);
  encoding = updateEncodingForShape(op, sharedEnc, existingTy);
  return TensorDescType::get(existingTy.getShape(), existingTy.getElementType(),
                             encoding);
}

struct UseInfo {
  TypedValue<TensorDescType> descriptor;
  Operation *use;
  Attribute desiredSharedEncoding;
  SmallVector<int64_t> shape;
  CGAEncodingAttr cgaLayout;
};

struct EncodingInfo {
  Attribute desiredEncoding;
  CGAEncodingAttr cgaLayout;
  // Shape may be different from the descriptor block shape for gather/scatter
  // use case
  SmallVector<int64_t> shape;
  bool forcedToDefault = false;

  bool operator==(const EncodingInfo &other) const {
    return desiredEncoding == other.desiredEncoding &&
           cgaLayout == other.cgaLayout &&
           forcedToDefault == other.forcedToDefault && shape == other.shape;
  }
};

} // namespace mlir::triton::gpu

template <> struct std::hash<ttg::EncodingInfo> {
  size_t operator()(const ttg::EncodingInfo &einfo) const {
    return llvm::hash_combine(einfo.desiredEncoding, einfo.cgaLayout,
                              einfo.forcedToDefault,
                              llvm::ArrayRef<int64_t>(einfo.shape));
  }
};

namespace mlir::triton::gpu {
//===----------------------------------------------------------------------===//
// AssignDescriptorMemoryLayouts
//===----------------------------------------------------------------------===//

const EncodingInfo *AssignDescriptorMemoryLayouts::internEncoding(
    std::unordered_set<EncodingInfo> &encodings, EncodingInfo info) {
  return &*encodings.insert(info).first;
}

EncodingInfo AssignDescriptorMemoryLayouts::combineEncodings(
    const EncodingInfo &lhs, const EncodingInfo &rhs, unsigned rank) {
  EncodingInfo result;
  // Always propagate forcedToDefault
  result.forcedToDefault = lhs.forcedToDefault || rhs.forcedToDefault;

  if (result.forcedToDefault)
    return result;

  if (lhs.shape.empty() || lhs.shape == rhs.shape)
    result.shape = rhs.shape;
  else if (rhs.shape.empty())
    result.shape = lhs.shape;
  else {
    assert(lhs.shape.size() == rhs.shape.size());
    auto rank = lhs.shape.size();
    result.shape.reserve(rank);
    for (int i = 0; i < rank; ++i)
      result.shape.push_back(std::min(lhs.shape[i], rhs.shape[i]));
  }

  SetVector<CGAEncodingAttr> cgaLayouts;
  if (lhs.cgaLayout)
    cgaLayouts.insert(lhs.cgaLayout);
  if (rhs.cgaLayout)
    cgaLayouts.insert(rhs.cgaLayout);

  auto getDefaultLayout = [&](CGAEncodingAttr encoding) {
    // The default layout puts all the CTAs in the last dimension
    // We do this as this function needs to be commutative for all encodings
    // This heuristic could be improved if needed
    auto ctx = encoding.getContext();
    auto kBlock = StringAttr::get(ctx, "block");
    auto dims = triton::standardOutDimNames(ctx, rank);
    auto numCTAs = encoding.getLinearLayout().getInDimSize(kBlock);
    LinearLayout llDefault;
    for (int i = 0; i < rank - 1; ++i) {
      llDefault *= LinearLayout::identity1D(1, kBlock, dims[i]);
    }
    llDefault *= LinearLayout::identity1D(numCTAs, kBlock, dims.back());
    return CGAEncodingAttr::get(ctx, llDefault);
  };

  switch (cgaLayouts.size()) {
  case 2:
    // if we find clashing CGALayouts, fallback to default
    result.cgaLayout = getDefaultLayout(lhs.cgaLayout);
    break;
  case 1:
    result.cgaLayout = cgaLayouts[0];
    break;
  default:
    break;
  }

  SetVector<Attribute> desiredEncodings;
  if (lhs.desiredEncoding)
    desiredEncodings.insert(lhs.desiredEncoding);
  if (rhs.desiredEncoding)
    desiredEncodings.insert(rhs.desiredEncoding);

  switch (desiredEncodings.size()) {
  case 2:
    // if we find clashing encodings, fallback to default
    result.forcedToDefault = true;
    break;
  case 1:
    result.desiredEncoding = desiredEncodings[0];
    break;
  default:
    break;
  }
  return result;
}

Attribute
AssignDescriptorMemoryLayouts::findLoadEncodingFromUsers(Operation *op) {
  // Check if there are any desired encodings available on the op
  if (auto attr = op->getDiscardableAttr("tt.desired_encoding")) {
    if (auto enc = dyn_cast<ttg::SharedEncodingTrait>(attr)) {
      if (isCompatibleSharedEncoding(enc))
        return enc;
    }
  }
  // Ignore multiple users and just pick the first compatible layout
  for (auto use : op->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(use)) {
      auto enc = alloc.getType().getEncoding();
      if (isCompatibleSharedEncoding(enc))
        return enc;
    } else if (auto store = dyn_cast<ttg::LocalStoreOp>(use)) {
      auto enc = store.getDst().getType().getEncoding();
      if (isCompatibleSharedEncoding(enc))
        return enc;
    }
  }
  return {};
}

std::optional<UseInfo>
AssignDescriptorMemoryLayouts::getUseInfo(Operation *op) {
  UseInfo info;
  info.use = op;
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    info.descriptor = load.getDesc();
    info.desiredSharedEncoding = findLoadEncodingFromUsers(op);
    auto encoding = info.desiredSharedEncoding ? info.desiredSharedEncoding
                                               : load.getType().getEncoding();
    info.cgaLayout = getCGALayout(encoding);
    auto shape = load.getResult().getType().getShape();
    auto rank = load.getDesc().getType().getShape().size();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  if (auto gather = dyn_cast<DescriptorGatherOp>(op)) {
    info.descriptor = gather.getDesc();
    info.desiredSharedEncoding = findLoadEncodingFromUsers(op);
    auto encoding = info.desiredSharedEncoding ? info.desiredSharedEncoding
                                               : gather.getType().getEncoding();
    info.cgaLayout = getCGALayout(encoding);
    auto shape = gather.getResult().getType().getShape();
    auto rank = gather.getDesc().getType().getShape().size();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  if (auto store = dyn_cast<DescriptorStoreLikeOpInterface>(op)) {
    info.descriptor = store.getDesc();
    auto encoding = store.getSrc().getType().getEncoding();
    info.cgaLayout = getCGALayout(encoding);
    auto shape = store.getSrc().getType().getShape();
    auto rank = store.getDesc().getType().getShape().size();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  return std::nullopt;
}

// Build fallback shared encoding with callback
Attribute AssignDescriptorMemoryLayouts::getFallbackSharedEncoding(
    RankedTensorType tensorType, ttg::CGAEncodingAttr cgaLayout,
    ArrayRef<int64_t> usageShape, unsigned numCTAs) {
  auto ctx = tensorType.getContext();
  SmallVector<unsigned> order;
  for (int i = tensorType.getRank() - 1; i >= 0; --i)
    order.push_back(i);

  ArrayRef<int64_t> shape =
      usageShape.empty() ? tensorType.getShape() : usageShape;
  if (!cgaLayout) {
    // Arbitrarily distribute along the last dim
    SmallVector<unsigned> ctasPerCGA(tensorType.getRank(), 1);
    ctasPerCGA.back() = numCTAs;
    cgaLayout = ttg::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA,
                                                      ctasPerCGA, order);
  } else if (cgaLayout.getRank() != tensorType.getRank())
    cgaLayout = updateCGALayoutForShape(cgaLayout, shape);

  return buildFallbackSharedEncoding(ctx, shape, order, cgaLayout,
                                     tensorType.getElementType());
}

// For each function compute shared memory encodings for all descriptors. The
// encodings are derived from the uses by applying findEncodingFromUsers on ops.
// The computed encoding information (EncodingInfo) is then propagated through
// a fixed point iteration to all descriptors in the function. A shared encoding
// is then fully materialized either using an existing shared encoding or by
// applying getFallbackSharedEncoding. We then apply updateEncodingForShape to
// adapt the encoding to the shape of the descriptor.
void AssignDescriptorMemoryLayouts::runOnFunction(FuncOp &func) {
  std::unordered_set<EncodingInfo> encodings;
  llvm::MapVector<TypedValue<TensorDescType>, const EncodingInfo *>
      valueToEncodingInfo;
  llvm::PriorityWorklist<TypedValue<triton::TensorDescType>> worklist;

  auto updateEncoding = [&](ArrayRef<Value> descValues, EncodingInfo info) {
    for (auto value : descValues) {
      auto typedVal = cast<TypedValue<TensorDescType>>(value);
      auto itr = valueToEncodingInfo.find(typedVal);
      if (itr != valueToEncodingInfo.end())
        info = combineEncodings(*itr->second, info,
                                typedVal.getType().getShape().size());
    }

    auto einfo = internEncoding(encodings, info);
    for (auto value : descValues) {
      auto typedVal = cast<TypedValue<TensorDescType>>(value);
      auto res = valueToEncodingInfo.try_emplace(typedVal, einfo);
      if (res.second) {
        worklist.insert(typedVal);
      } else if (res.first->second != einfo) {
        res.first->second = einfo;
        worklist.insert(typedVal);
      }
    }
  };

  // 1. Set seed values from either TMA ops, or device function boundaries for
  // which we fallback to default encoding
  auto isKernel = triton::isKernel(func);
  for (auto blockArg : func.getBlocks().front().getArguments())
    if (auto desc = dyn_cast<TypedValue<TensorDescType>>(blockArg))
      updateEncoding({desc},
                     EncodingInfo{{}, {}, {}, /*forcedToDefault=*/!isKernel});

  func.walk([&](Operation *op) {
    if (auto info = getUseInfo(op)) {
      updateEncoding(info->descriptor,
                     EncodingInfo{info->desiredSharedEncoding, info->cgaLayout,
                                  info->shape});
    } else {
      bool forcedToDefault =
          isa<CallOp, ReturnOp, ttng::ReinterpretTensorDescOp>(op);
      auto einfo =
          internEncoding(encodings, EncodingInfo{{}, {}, {}, forcedToDefault});

      auto setEncoding = [&](Value v) {
        auto typedVal = cast<TypedValue<TensorDescType>>(v);
        valueToEncodingInfo.try_emplace(typedVal, einfo);
        if (forcedToDefault)
          worklist.insert(typedVal);
      };
      for (auto result : op->getResults())
        if (auto desc = dyn_cast<TypedValue<TensorDescType>>(result))
          setEncoding(desc);

      for (auto arg : op->getOperands())
        if (auto desc = dyn_cast<TypedValue<TensorDescType>>(arg))
          setEncoding(desc);
    }
  });

  // 2. Propagate encoding info through the graph until fixed point
  while (!worklist.empty()) {
    auto desc = worklist.pop_back_val();

    // Propagate to users
    for (OpOperand &use : desc.getUses()) {
      auto op = use.getOwner();
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        auto offset = 3 * isa<scf::ForOp>(op);
        auto vals = getTiedArgs(op, use.getOperandNumber() - offset);
        updateEncoding(vals, EncodingInfo{});
      } else if (isa<scf::YieldOp>(op)) {
        auto vals = getTiedArgs(op->getParentOp(), use.getOperandNumber());
        updateEncoding(vals, EncodingInfo{});
      }
    }

    // Propagate to defining ops
    if (auto opResult = dyn_cast<OpResult>(desc)) {
      auto definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto vals = getTiedArgs(definingOp, opResult.getResultNumber());
        updateEncoding(vals, EncodingInfo{});
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(desc)) {
      auto parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        auto offset = isa<scf::ForOp>(parentOp);
        auto vals = getTiedArgs(parentOp, blockArg.getArgNumber() - offset);
        updateEncoding(vals, EncodingInfo{});
      }
    }
  }

  // 3. Transfer propagated encodings into the graph
  auto ctx = func.getContext();
  auto numCTAs = triton::gpu::lookupNumCTAs(func);
  for (auto &[desc, einfo] : valueToEncodingInfo) {
    auto existingTy = desc.getType().getBlockType();
    Attribute newEncoding;
    if (einfo->desiredEncoding) {
      newEncoding = einfo->desiredEncoding;
    } else if (einfo->forcedToDefault) {
      newEncoding = getFallbackSharedEncoding(existingTy, {}, {}, numCTAs);
    } else {
      newEncoding = getFallbackSharedEncoding(existingTy, einfo->cgaLayout,
                                              einfo->shape, numCTAs);
    }
    desc.setType(getTensorDescTypeWithEncoding(desc.getDefiningOp(), existingTy,
                                               newEncoding));
  }

  SmallVector<Type> argTys(func.getBlocks().front().getArgumentTypes());
  SmallVector<Type> resultTys(func.getResultTypes());
  for (auto [i, resultTy] : llvm::enumerate(resultTys)) {
    if (auto descTy = dyn_cast<TensorDescType>(resultTy)) {
      auto encoding =
          getFallbackSharedEncoding(descTy.getBlockType(), {}, {}, numCTAs);
      resultTys[i] = getTensorDescTypeWithEncoding(
          nullptr, descTy.getBlockType(), encoding);
    }
  }
  func.setFunctionType(FunctionType::get(ctx, argTys, resultTys));
}

void AssignDescriptorMemoryLayouts::assignMemoryLayouts(ModuleOp &mod) {
  for (auto &op : *mod.getBody()) {
    if (auto func = dyn_cast<FuncOp>(&op)) {
      runOnFunction(func);
    }
  }
}
} // namespace mlir::triton::gpu
