#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include <algorithm>
#include <optional>

namespace mlir::triton::instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONINSTRUMENTGLOBALSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

static constexpr const char kGSanGlobalStateArgAttr[] = "tti.gsan_global_state";
static constexpr const char kDisableSetMaxRegisterAttr[] =
    "tti.disable_setmaxregister";

struct DescriptorInfo {
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
};

static void setTMAPtrAxisHints(OpBuilder &builder, Value ptr) {
  auto ptrTy = cast<RankedTensorType>(ptr.getType());
  auto elemTy = cast<tt::PointerType>(ptrTy.getElementType()).getPointeeType();

  Operation *def = ptr.getDefiningOp();
  if (!def)
    return;

  auto rank = ptrTy.getRank();
  SmallVector<int32_t> contiguity(rank, 1);
  contiguity.back() = ptrTy.getShape().back();
  SmallVector<int32_t> divisibility(rank, 1);
  divisibility.back() = 16;
  auto attrTy = RankedTensorType::get({rank}, builder.getI32Type());
  def->setDiscardableAttr("tt.contiguity",
                          DenseIntElementsAttr::get(attrTy, contiguity));
  def->setDiscardableAttr("tt.divisibility",
                          DenseIntElementsAttr::get(attrTy, divisibility));
}

static Value castToI64(OpBuilder &builder, Location loc, Value value) {
  if (value.getType().isInteger(64))
    return value;
  return builder.createOrFold<arith::ExtSIOp>(loc, builder.getI64Type(), value);
}

static SmallVector<Value> castToI64(OpBuilder &builder, Location loc,
                                    ValueRange values) {
  SmallVector<Value> result;
  result.reserve(values.size());
  for (Value value : values)
    result.push_back(castToI64(builder, loc, value));
  return result;
}

static ttg::BlockedEncodingAttr
getInstrumentationEncoding(OpBuilder &builder, ArrayRef<int64_t> shape,
                           Type elemType) {
  int numWarps = ttg::lookupNumWarps(builder.getInsertionBlock()->getParent());
  int threadsPerWarp = ttg::lookupThreadsPerWarp(builder);
  int numCTAs = ttg::lookupNumCTAs(builder.getInsertionBlock()->getParentOp());
  auto base = ttg::getDefaultBlockedEncoding(builder.getContext(), shape,
                                             numWarps, threadsPerWarp, numCTAs);
  SmallVector<unsigned> order = llvm::to_vector(base.getOrder());
  SmallVector<unsigned> warpsPerCTA = llvm::to_vector(base.getWarpsPerCTA());
  SmallVector<unsigned> sizePerThread(shape.size(), 1);
  unsigned elemBits = elemType.getIntOrFloatBitWidth();
  unsigned maxElems = std::max(128u / elemBits, 1u);
  if (!order.empty()) {
    unsigned dim = order.front();
    // Distribute last dim to maximize contiguity within a thread
    if (order.size() > 1 && warpsPerCTA[dim] > 1) {
      warpsPerCTA[order[1]] *= warpsPerCTA[dim];
      warpsPerCTA[dim] = 1;
    }

    auto threadsOnDim = base.getThreadsPerWarp()[dim] * warpsPerCTA[dim];
    auto numUniqueElems = ceil(static_cast<unsigned>(shape[dim]), threadsOnDim);
    sizePerThread[dim] = std::min(maxElems, numUniqueElems);
  }
  return ttg::BlockedEncodingAttr::get(builder.getContext(), sizePerThread,
                                       base.getThreadsPerWarp(), warpsPerCTA,
                                       order, base.getCGALayout());
}

static Value expandAllSlicedDims(OpBuilder &builder, Location loc,
                                 Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto sliceEncoding = dyn_cast<ttg::SliceEncodingAttr>(type.getEncoding());
  while (sliceEncoding) {
    tensor = expandOuterSlicedDim(builder, loc, tensor);
    type = cast<RankedTensorType>(tensor.getType());
    sliceEncoding = dyn_cast<ttg::SliceEncodingAttr>(type.getEncoding());
  }
  return tensor;
}

static DescriptorInfo getDescriptorInfo(Value desc, OpBuilder &builder) {
  if (!isa<tt::TensorDescType>(desc.getType())) {
    std::string msg;
    llvm::raw_string_ostream stream(msg);
    stream << "GSan: Unsupported descriptor type" << desc.getType();
    llvm::report_fatal_error(msg.c_str());
  }
  auto descTy = cast<tt::TensorDescType>(desc.getType());

  auto elemTy = descTy.getSignlessBlockType().getElementType();
  auto basePtrTy = tt::getPointerType(elemTy);
  unsigned rank = descTy.getBlockType().getRank();
  SmallVector<Type> resultTypes;
  resultTypes.reserve(1 + 2 * rank);
  resultTypes.push_back(basePtrTy);
  resultTypes.append(rank, builder.getI64Type());
  resultTypes.append(rank, builder.getI64Type());

  auto info = ExperimentalGSanTensorDescInfoOp::create(builder, desc.getLoc(),
                                                       resultTypes, desc);
  auto results = info->getResults();

  DescriptorInfo descriptorInfo;
  descriptorInfo.base = results.front();
  descriptorInfo.shape.assign(results.begin() + 1, results.begin() + 1 + rank);
  descriptorInfo.strides.assign(results.begin() + 1 + rank, results.end());
  return descriptorInfo;
}

static Value createExpandedOffsetRange(OpBuilder &builder, Location loc,
                                       RankedTensorType fullI64Type,
                                       Value offset, unsigned dim) {
  auto fullEncoding =
      cast<ttg::DistributedEncodingTrait>(fullI64Type.getEncoding());
  auto sliceEncoding = getSingleDimSliceEncoding(fullEncoding, dim);
  int64_t dimSize = fullI64Type.getShape()[dim];

  auto sliceI32Type =
      RankedTensorType::get({dimSize}, builder.getI32Type(), sliceEncoding);
  auto sliceI64Type =
      RankedTensorType::get({dimSize}, builder.getI64Type(), sliceEncoding);

  Value range = tt::MakeRangeOp::create(builder, loc, sliceI32Type, 0, dimSize);
  Value rangeI64 = arith::ExtSIOp::create(builder, loc, sliceI64Type, range);
  Value offsetI64 = castToI64(builder, loc, offset);
  Value offsetSplat =
      tt::SplatOp::create(builder, loc, sliceI64Type, offsetI64);
  Value result =
      arith::AddIOp::create(builder, loc, sliceI64Type, offsetSplat, rangeI64);
  result = expandAllSlicedDims(builder, loc, result);
  if (cast<RankedTensorType>(result.getType()).getShape() !=
      fullI64Type.getShape()) {
    result = tt::BroadcastOp::create(builder, loc, fullI64Type, result);
  }
  return result;
}

static Value convertAndBroadcast(OpBuilder &builder, Location loc, Value tensor,
                                 int dim, RankedTensorType dstType) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto encoding = cast<ttg::DistributedEncodingTrait>(dstType.getEncoding());
  auto sliceEncoding = getSingleDimSliceEncoding(encoding, dim);
  auto sliceType = RankedTensorType::get(
      tensorType.getShape(), tensorType.getElementType(), sliceEncoding);
  tensor = ttg::ConvertLayoutOp::create(builder, loc, sliceType, tensor);
  tensor = expandAllSlicedDims(builder, loc, tensor);
  if (cast<RankedTensorType>(tensor.getType()).getShape() != dstType.getShape())
    tensor = tt::BroadcastOp::create(builder, loc, dstType, tensor);
  return tensor;
}

static Value createMaskFromRanges(OpBuilder &builder, Location loc,
                                  const DescriptorInfo &desc,
                                  ArrayRef<Value> offsetRanges,
                                  RankedTensorType fullI64Type) {
  auto maskType = RankedTensorType::get(
      fullI64Type.getShape(), builder.getI1Type(), fullI64Type.getEncoding());
  Value zero = createConstIntTensor(builder, loc, 0, fullI64Type,
                                    /*isSigned=*/true);

  Value mask;
  for (auto [dim, offsets] : llvm::enumerate(offsetRanges)) {
    Value upperBound =
        tt::SplatOp::create(builder, loc, fullI64Type, desc.shape[dim]);
    Value lower = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                        offsets, zero);
    Value upper = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        offsets, upperBound);
    Value dimMask = arith::AndIOp::create(builder, loc, lower, upper);
    dimMask = cast<RankedTensorType>(dimMask.getType()) == maskType
                  ? dimMask
                  : tt::BroadcastOp::create(builder, loc, maskType, dimMask);
    mask = mask ? arith::AndIOp::create(builder, loc, mask, dimMask) : dimMask;
  }
  return mask;
}

static Value createPtrFromRanges(OpBuilder &builder, Location loc,
                                 const DescriptorInfo &desc,
                                 ArrayRef<Value> offsetRanges,
                                 RankedTensorType fullI64Type) {
  auto ptrTensorType = RankedTensorType::get(
      fullI64Type.getShape(), desc.base.getType(), fullI64Type.getEncoding());
  Value ptr = tt::SplatOp::create(builder, loc, ptrTensorType, desc.base);
  for (auto [dim, offsets] : llvm::enumerate(offsetRanges)) {
    Value stride =
        tt::SplatOp::create(builder, loc, fullI64Type, desc.strides[dim]);
    Value offsetWithStride =
        arith::MulIOp::create(builder, loc, fullI64Type, offsets, stride);
    ptr = tt::AddPtrOp::create(builder, loc, ptrTensorType, ptr,
                               offsetWithStride);
  }
  return ptr;
}

static std::pair<Value, Value>
createTiledAccess(OpBuilder &builder, Location loc, const DescriptorInfo &desc,
                  ArrayRef<int64_t> blockShape, ValueRange offsets,
                  std::optional<Value> pred) {
  auto encoding = getInstrumentationEncoding(
      builder, blockShape,
      cast<tt::PointerType>(desc.base.getType()).getPointeeType());
  auto fullI64Type =
      RankedTensorType::get(blockShape, builder.getI64Type(), encoding);

  SmallVector<Value> offsetRanges;
  offsetRanges.reserve(offsets.size());
  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    offsetRanges.push_back(
        createExpandedOffsetRange(builder, loc, fullI64Type, offset, dim));
  }

  Value ptr =
      createPtrFromRanges(builder, loc, desc, offsetRanges, fullI64Type);
  Value mask =
      createMaskFromRanges(builder, loc, desc, offsetRanges, fullI64Type);
  if (pred) {
    auto maskType = cast<RankedTensorType>(mask.getType());
    Value predTensor = tt::SplatOp::create(builder, loc, maskType, *pred);
    mask = arith::AndIOp::create(builder, loc, mask, predTensor);
  }
  setTMAPtrAxisHints(builder, ptr);
  return std::make_pair(ptr, mask);
}

static std::pair<Value, Value> createGatherScatterAccess(
    OpBuilder &builder, Location loc, const DescriptorInfo &desc,
    ArrayRef<int64_t> blockShape, Value xOffsets, Value yOffset) {
  auto encoding = getInstrumentationEncoding(
      builder, blockShape,
      cast<tt::PointerType>(desc.base.getType()).getPointeeType());
  auto fullI64Type =
      RankedTensorType::get(blockShape, builder.getI64Type(), encoding);

  auto xOffsetsTy = cast<RankedTensorType>(xOffsets.getType());
  auto xOffsetsI64Ty = RankedTensorType::get(
      xOffsetsTy.getShape(), builder.getI64Type(), xOffsetsTy.getEncoding());
  Value xOffsetsI64 =
      arith::ExtSIOp::create(builder, loc, xOffsetsI64Ty, xOffsets);
  Value xRange =
      convertAndBroadcast(builder, loc, xOffsetsI64, /*dim=*/0, fullI64Type);
  Value yRange =
      createExpandedOffsetRange(builder, loc, fullI64Type, yOffset, /*dim=*/1);
  SmallVector<Value> offsetRanges = {xRange, yRange};
  auto ptrs =
      createPtrFromRanges(builder, loc, desc, offsetRanges, fullI64Type);
  auto mask =
      createMaskFromRanges(builder, loc, desc, offsetRanges, fullI64Type);
  setTMAPtrAxisHints(builder, ptrs);
  return std::make_pair(ptrs, mask);
}

static void instrumentAsyncTMALoad(ttng::AsyncTMACopyGlobalToLocalOp op) {
  if (isa<ttng::TensorDescIm2ColType>(op.getDesc().getType()))
    return;

  OpBuilder builder(op);
  auto desc = getDescriptorInfo(op.getDesc(), builder);

  auto offsets = castToI64(builder, op.getLoc(), op.getCoord());
  auto access = createTiledAccess(builder, op.getLoc(), desc,
                                  op.getResult().getType().getShape(), offsets,
                                  op.getPred());
  ExperimentalGSanTensorAccessOp::create(builder, op.getLoc(), access.first,
                                         access.second, /*isStore=*/false);
}

static void instrumentAsyncTMAStore(Operation *op, Value descValue,
                                    ArrayRef<int64_t> blockShape,
                                    ValueRange coords) {
  OpBuilder builder(op);
  auto desc = getDescriptorInfo(descValue, builder);

  auto offsets = castToI64(builder, op->getLoc(), coords);
  auto access = createTiledAccess(builder, op->getLoc(), desc, blockShape,
                                  offsets, std::nullopt);
  ExperimentalGSanTensorAccessOp::create(builder, op->getLoc(), access.first,
                                         access.second, /*isStore=*/true);
}

static void instrumentAsyncTMAGather(ttng::AsyncTMAGatherOp op) {
  OpBuilder builder(op);
  auto desc = getDescriptorInfo(op.getDesc(), builder);

  auto access = createGatherScatterAccess(builder, op.getLoc(), desc,
                                          op.getResult().getType().getShape(),
                                          op.getXOffsets(), op.getYOffset());
  auto maskType = cast<RankedTensorType>(access.second.getType());
  Value predTensor =
      tt::SplatOp::create(builder, op.getLoc(), maskType, op.getPred());
  Value mask =
      arith::AndIOp::create(builder, op.getLoc(), access.second, predTensor);
  ExperimentalGSanTensorAccessOp::create(builder, op.getLoc(), access.first,
                                         mask, /*isStore=*/false);
}

static void instrumentAsyncTMAScatter(ttng::AsyncTMAScatterOp op) {
  OpBuilder builder(op);
  auto desc = getDescriptorInfo(op.getDesc(), builder);

  auto access = createGatherScatterAccess(builder, op.getLoc(), desc,
                                          op.getSrc().getType().getShape(),
                                          op.getXOffsets(), op.getYOffset());
  ExperimentalGSanTensorAccessOp::create(builder, op.getLoc(), access.first,
                                         access.second, /*isStore=*/true);
}

class GlobalSanitizerPass
    : public impl::TritonInstrumentGlobalSanitizerBase<GlobalSanitizerPass> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    OpBuilder builder(module);
    Type gsanStatePtrTy = tt::PointerType::get(builder.getI8Type(), 1);
    DenseSet<StringRef> calledFuncs;
    module.walk(
        [&](tt::CallOp callOp) { calledFuncs.insert(callOp.getCallee()); });

    SmallVector<tt::FuncOp> funcs;
    module.walk([&](tt::FuncOp func) { funcs.push_back(func); });
    for (tt::FuncOp func : funcs) {
      auto funcTy = func.getFunctionType();
      SmallVector<Type> inputTys(funcTy.getInputs().begin(),
                                 funcTy.getInputs().end());
      inputTys.push_back(gsanStatePtrTy);
      func.setType(FunctionType::get(module.getContext(), inputTys,
                                     funcTy.getResults()));

      func.getBody().addArgument(gsanStatePtrTy, func.getLoc());
      SmallVector<Attribute> newArgAttrs;
      if (auto argAttrs = func.getAllArgAttrs())
        newArgAttrs.append(argAttrs.begin(), argAttrs.end());
      while (newArgAttrs.size() < func.getNumArguments()) {
        newArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
      }
      if (!newArgAttrs.empty())
        func.setAllArgAttrs(newArgAttrs);
      func.setArgAttr(func.getNumArguments() - 1, kGSanGlobalStateArgAttr,
                      builder.getUnitAttr());

      bool isEntry = !calledFuncs.contains(func.getSymName());
      if (isEntry) {
        OpBuilder b(&func.front(), func.front().begin());
        ExperimentalGSanInitOp::create(b, func.getLoc());
      }
    }

    SmallVector<tt::CallOp> callOps;
    module.walk([&](tt::CallOp op) { callOps.push_back(op); });
    for (tt::CallOp callOp : callOps) {
      auto caller = callOp->getParentOfType<tt::FuncOp>();
      assert(caller && caller.getNumArguments() > 0 &&
             "expected triton.call to be nested under a Triton function");

      SmallVector<Value> operands(callOp.getOperands().begin(),
                                  callOp.getOperands().end());
      operands.push_back(caller.getArgument(caller.getNumArguments() - 1));

      OpBuilder b(callOp);
      auto newCallOp =
          tt::CallOp::create(b, callOp.getLoc(), callOp.getCallee(),
                             callOp.getResultTypes(), operands);
      newCallOp->setAttrs(callOp->getAttrs());
      callOp->replaceAllUsesWith(newCallOp->getResults());
      callOp.erase();
    }

    module.walk([&](Operation *op) {
      OpBuilder b(op);
      mlir::TypeSwitch<Operation *>(op)
          .Case([&](tt::LoadOp op) {
            ExperimentalGSanTensorAccessOp::create(
                b, op.getLoc(), op.getPtr(), op.getMask(), /*isStore=*/false);
          })
          .Case([&](tt::StoreOp op) {
            ExperimentalGSanTensorAccessOp::create(
                b, op.getLoc(), op.getPtr(), op.getMask(), /*isStore=*/true);
          })
          .Case([&](ttg::AsyncCopyGlobalToLocalOp op) {
            ExperimentalGSanTensorAccessOp::create(
                b, op.getLoc(), op.getSrc(), op.getMask(), /*isStore=*/false);
          })
          .Case([&](ttng::AsyncTMACopyGlobalToLocalOp op) {
            instrumentAsyncTMALoad(op);
          })
          .Case(
              [&](ttng::AsyncTMAGatherOp op) { instrumentAsyncTMAGather(op); })
          .Case([&](ttng::AsyncTMACopyLocalToGlobalOp op) {
            instrumentAsyncTMAStore(op, op.getDesc(),
                                    op.getSrc().getType().getShape(),
                                    op.getCoord());
          })
          .Case([&](ttng::AsyncTMAReduceOp op) {
            // FIXME: This is just plain wrong. TMA reduce is atomic.
            instrumentAsyncTMAStore(op, op.getDesc(),
                                    op.getSrc().getType().getShape(),
                                    op.getCoord());
          })
          .Case([&](ttng::AsyncTMAScatterOp op) {
            instrumentAsyncTMAScatter(op);
          });
    });

    module.walk([&](ttg::WarpSpecializeOp op) {
      op->setAttr(kDisableSetMaxRegisterAttr, builder.getUnitAttr());
    });
  }
};

} // namespace

} // namespace mlir::triton::instrument
