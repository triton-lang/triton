#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/TMAUtilities.h"
#include "llvm/ADT/PriorityWorklist.h"
#include <algorithm>
#include <unordered_set>

namespace ttg = mlir::triton::gpu;

namespace {

struct UseInfo {
  TypedValue<TensorDescType> descriptor;
  Operation *use;
  Attribute desiredSharedEncoding;
  SmallVector<int64_t> shape;
  ttg::CTALayoutAttr ctaLayout;
};

static bool isTMACompatibleEncoding(Attribute enc) {
  if (auto nvmma = dyn_cast<ttg::NVMMASharedEncodingAttr>(enc)) {
    return !nvmma.getTransposed();
  }
  return false;
}

Attribute findLoadEncodingFromUsers(Operation *op) {
  // Ignore multiple users and just pick the first compatible layout
  for (auto use : op->getUsers()) {
    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(use)) {
      auto enc = alloc.getType().getEncoding();
      if (isTMACompatibleEncoding(enc))
        return enc;
    } else if (auto store = dyn_cast<ttg::LocalStoreOp>(use)) {
      auto enc = store.getDst().getType().getEncoding();
      if (isTMACompatibleEncoding(enc))
        return enc;
    }
  }
  return {};
}

SmallVector<int64_t> expandToRank(ArrayRef<int64_t> shape, int rank) {
  SmallVector<int64_t> result(rank, 1);
  assert(shape.size() <= rank);
  auto rankDiff = rank - shape.size();
  std::copy(shape.begin(), shape.end(), result.begin() + rankDiff);
  return result;
}

std::optional<UseInfo> getUseInfo(Operation *op) {
  UseInfo info;
  info.use = op;
  if (auto load = dyn_cast<DescriptorLoadOp>(op)) {
    info.descriptor = load.getDesc();
    info.desiredSharedEncoding = findLoadEncodingFromUsers(op);
    auto encoding = info.desiredSharedEncoding ? info.desiredSharedEncoding
                                               : load.getType().getEncoding();
    info.ctaLayout = ttg::getCTALayout(encoding);
    auto shape = load.getResult().getType().getShape();
    auto rank = load.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  if (auto gather = dyn_cast<DescriptorGatherOp>(op)) {
    info.descriptor = gather.getDesc();
    info.desiredSharedEncoding = findLoadEncodingFromUsers(op);
    auto encoding = info.desiredSharedEncoding ? info.desiredSharedEncoding
                                               : gather.getType().getEncoding();
    info.ctaLayout = ttg::getCTALayout(encoding);
    auto shape = gather.getResult().getType().getShape();
    auto rank = gather.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  if (auto store = dyn_cast<DescriptorStoreLikeOpInterface>(op)) {
    info.descriptor = store.getDesc();
    auto encoding = store.getSrc().getType().getEncoding();
    info.ctaLayout = ttg::getCTALayout(encoding);
    auto shape = store.getSrc().getType().getShape();
    auto rank = store.getDesc().getType().getBlockType().getRank();
    info.shape = expandToRank(shape, rank);
    return info;
  }
  return std::nullopt;
}

struct EncodingInfo {
  Attribute desiredEncoding;
  ttg::CTALayoutAttr ctaLayout;
  // Shape may be different from the descriptor block shape for gather/scatter
  // use case
  SmallVector<int64_t> shape;
  bool forcedToDefault = false;

  bool operator==(const EncodingInfo &other) const {
    return desiredEncoding == other.desiredEncoding &&
           ctaLayout == other.ctaLayout &&
           forcedToDefault == other.forcedToDefault && shape == other.shape;
  }
};

} // namespace

template <> struct std::hash<EncodingInfo> {
  size_t operator()(const EncodingInfo &einfo) const {
    return llvm::hash_combine(einfo.desiredEncoding, einfo.ctaLayout,
                              einfo.forcedToDefault,
                              ArrayRef<int64_t>(einfo.shape));
  }
};

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUOPTIMIZEDESCRIPTORENCODINGPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

const EncodingInfo *internEncoding(std::unordered_set<EncodingInfo> &encodings,
                                   EncodingInfo info) {
  return &*encodings.insert(info).first;
}

EncodingInfo combineEncodings(const EncodingInfo &lhs, const EncodingInfo &rhs,
                              unsigned rank) {
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

  SetVector<ttg::CTALayoutAttr> ctaLayouts;
  if (lhs.ctaLayout)
    ctaLayouts.insert(lhs.ctaLayout);
  if (rhs.ctaLayout)
    ctaLayouts.insert(rhs.ctaLayout);

  switch (ctaLayouts.size()) {
  case 2:
    // if we find clashing CTALayouts, fallback to default
    result.ctaLayout =
        ttg::CTALayoutAttr::getDefault(lhs.ctaLayout.getContext(), rank);
    break;
  case 1:
    result.ctaLayout = ctaLayouts[0];
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

Attribute getFallbackSharedEncoding(RankedTensorType tensorType,
                                    ttg::CTALayoutAttr ctaLayout,
                                    ArrayRef<int64_t> usageShape) {
  auto ctx = tensorType.getContext();
  SmallVector<unsigned> order;
  for (int i = tensorType.getRank() - 1; i >= 0; --i)
    order.push_back(i);

  ArrayRef<int64_t> shape =
      usageShape.empty() ? tensorType.getShape() : usageShape;
  if (!ctaLayout)
    ctaLayout = ttg::CTALayoutAttr::getDefault(ctx, tensorType.getRank());
  else if (ctaLayout.getRank() != tensorType.getRank())
    ctaLayout = updateCTALayoutForShape(ctaLayout, shape);

  return ttg::NVMMASharedEncodingAttr::get(ctx, shape, order, ctaLayout,
                                           tensorType.getElementType(),
                                           /*fp4Padded*/ false);
}

TensorDescType getTensorDescTypeWithEncoding(Operation *op,
                                             RankedTensorType existingTy,
                                             Attribute encoding) {
  auto sharedEnc = cast<triton::gpu::SharedEncodingTrait>(encoding);
  encoding = updateEncodingForShape(op, sharedEnc, existingTy);
  auto blockTy = RankedTensorType::get(existingTy.getShape(),
                                       existingTy.getElementType(), encoding);
  return TensorDescType::get(existingTy.getContext(), blockTy);
}

void assignMemoryLayouts(FuncOp &func) {
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
                                typedVal.getType().getBlockType().getRank());
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
                     EncodingInfo{info->desiredSharedEncoding, info->ctaLayout,
                                  info->shape});
    } else {
      bool forcedToDefault = isa<CallOp, ReturnOp, ReinterpretTensorDescOp>(op);
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
  for (auto &[desc, einfo] : valueToEncodingInfo) {
    auto existingTy = desc.getType().getBlockType();
    Attribute newEncoding;
    if (einfo->desiredEncoding) {
      newEncoding = einfo->desiredEncoding;
    } else if (einfo->forcedToDefault) {
      newEncoding = getFallbackSharedEncoding(existingTy, {}, {});
    } else {
      newEncoding =
          getFallbackSharedEncoding(existingTy, einfo->ctaLayout, einfo->shape);
    }
    desc.setType(getTensorDescTypeWithEncoding(desc.getDefiningOp(), existingTy,
                                               newEncoding));
  }

  SmallVector<Type> argTys(func.getBlocks().front().getArgumentTypes());
  SmallVector<Type> resultTys(func.getResultTypes());
  for (auto [i, resultTy] : llvm::enumerate(resultTys)) {
    if (auto descTy = dyn_cast<TensorDescType>(resultTy)) {
      auto encoding = getFallbackSharedEncoding(descTy.getBlockType(), {}, {});
      resultTys[i] = getTensorDescTypeWithEncoding(
          nullptr, descTy.getBlockType(), encoding);
    }
  }
  func.setFunctionType(FunctionType::get(ctx, argTys, resultTys));
}

void assignMemoryLayouts(ModuleOp &mod) {
  for (auto &op : *mod.getBody()) {
    if (auto func = dyn_cast<FuncOp>(&op)) {
      assignMemoryLayouts(func);
    }
  }
}

} // anonymous namespace

class TritonNvidiaGPUOptimizeDescriptorEncodingPass
    : public impl::TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
          TritonNvidiaGPUOptimizeDescriptorEncodingPass> {
public:
  using BaseT = TritonNvidiaGPUOptimizeDescriptorEncodingPassBase<
      TritonNvidiaGPUOptimizeDescriptorEncodingPass>;
  using BaseT::BaseT;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    assignMemoryLayouts(m);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
