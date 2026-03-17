#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <cassert>

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONINSTRUMENTFPSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

static bool isValueAvailableInScope(Value value, Region *scope) {
  if (!scope)
    return false;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    Region *argRegion = arg.getOwner()->getParent();
    return argRegion == scope || scope->isAncestor(argRegion);
  }
  if (Operation *def = value.getDefiningOp()) {
    Region *defRegion = def->getParentRegion();
    return defRegion == scope || scope->isAncestor(defRegion);
  }
  return false;
}

constexpr int64_t kTileM = 8;
constexpr int64_t kTileN = 8;

enum class UnaryOpId : uint64_t {
  Exp = 0,
  Log,
  Exp2,
  Log2,
  Cos,
  Sin,
  Sqrt,
  Rsqrt,
  Erf,
  Floor,
  Ceil,
  PreciseSqrt,
  DivInv,
};

constexpr uint64_t getUnaryOpId(UnaryOpId opId) {
  return static_cast<uint64_t>(opId);
}

// ------------------------------------------------------------
// Scratch memory management
// ------------------------------------------------------------

ttg::BlockedEncodingAttr getOptimizedBlockedEncoding(PatternRewriter &rewriter,
                                                     ArrayRef<int64_t> shape,
                                                     Type elemType) {
  int numWarps = ttg::lookupNumWarps(rewriter.getInsertionBlock()->getParent());
  int threadsPerWarp = ttg::lookupThreadsPerWarp(rewriter);
  int numCTAs = ttg::lookupNumCTAs(rewriter.getInsertionBlock()->getParentOp());
  auto base = ttg::getDefaultBlockedEncoding(rewriter.getContext(), shape,
                                             numWarps, threadsPerWarp, numCTAs);
  SmallVector<unsigned> order = llvm::to_vector(base.getOrder());
  SmallVector<unsigned> sizePerThread(shape.size(), 1);
  unsigned elemBits = elemType.getIntOrFloatBitWidth();
  unsigned maxElems = std::max(128u / elemBits, 1u);
  if (!order.empty()) {
    unsigned dim = order.front();
    sizePerThread[dim] =
        static_cast<unsigned>(std::min<int64_t>(shape[dim], maxElems));
  }
  return ttg::BlockedEncodingAttr::get(
      rewriter.getContext(), sizePerThread, base.getThreadsPerWarp(),
      base.getWarpsPerCTA(), order, base.getCGALayout());
}

struct ScratchInfo {
  Value ptr;
  RankedTensorType tensorType;
};

class TmemScratchManager {
public:
  ttg::BlockedEncodingAttr getScratchEncoding(PatternRewriter &rewriter,
                                              Value memdesc,
                                              ttg::MemDescType memTy) {
    return getOptimizedBlockedEncoding(rewriter, memTy.getShape(),
                                       memTy.getElementType());
  }

  static Value castToI32(PatternRewriter &rewriter, Location loc, Value value) {
    auto i32Ty = rewriter.getI32Type();
    auto ty = value.getType();
    if (ty == i32Ty)
      return value;
    if (ty.isIndex())
      return arith::IndexCastOp::create(rewriter, loc, i32Ty, value);
    if (auto intTy = dyn_cast<IntegerType>(ty)) {
      if (intTy.getWidth() > 32)
        return arith::TruncIOp::create(rewriter, loc, i32Ty, value);
      return arith::ExtSIOp::create(rewriter, loc, i32Ty, value);
    }
    return Value();
  }

  std::optional<ScratchInfo>
  getOrCreate(Value memdesc, PatternRewriter &rewriter, Region *scope) {
    if (auto arg = dyn_cast<BlockArgument>(memdesc)) {
      if (auto wsPartitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(
              arg.getOwner()->getParentOp())) {
        auto capture = wsPartitions.getExplicitCaptures()[arg.getArgNumber()];
        return getOrCreate(capture, rewriter, scope);
      }
      if (auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        unsigned argNum = arg.getArgNumber();
        if (argNum == 0)
          return std::nullopt;
        Value init = forOp.getInitArgs()[argNum - 1];
        return getOrCreate(init, rewriter, scope);
      }
      return std::nullopt;
    }

    auto memTy = dyn_cast<ttg::MemDescType>(memdesc.getType());
    if (!memTy || !isa<ttng::TensorMemorySpaceAttr>(memTy.getMemorySpace())) {
      return std::nullopt;
    }

    if (auto alloc = memdesc.getDefiningOp<ttng::TMEMAllocOp>()) {
      auto it = scratchMap.find(memdesc);
      if (it != scratchMap.end()) {
        auto itRegion = it->second.find(scope);
        if (itRegion != it->second.end()) {
          if (itRegion->second.ptr && itRegion->second.ptr.getType())
            return itRegion->second;
          it->second.erase(itRegion);
        }
      }

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(alloc);
      auto loc = alloc.getLoc();
      auto layout = getScratchEncoding(rewriter, memdesc, memTy);
      auto tensorTy = RankedTensorType::get(memTy.getShape(),
                                            memTy.getElementType(), layout);

      int64_t elSize = memTy.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t alignment = std::max<int64_t>(elSize, 16);
      int64_t sizeInBytes = product(memTy.getShape()) * elSize;
      auto ptrTy = triton::getPointerType(memTy.getElementType());
      auto allocOp = createThirdPartyScratchAlloc(rewriter, loc, ptrTy,
                                                  sizeInBytes, alignment);
      allocOp->setDiscardableAttr("tt.divisibility",
                                  rewriter.getI64IntegerAttr(alignment));
      Value ptr = allocOp.getResult();

      if (Value init = alloc.getSrc()) {
        auto initTy = cast<RankedTensorType>(init.getType());
        if (!createStoreScratchMemory(rewriter, loc, ptr, init, initTy))
          return std::nullopt;
      }

      ptr = remapToScope(ptr, rewriter, scope, loc);

      ScratchInfo info{ptr, tensorTy};
      scratchMap[memdesc][scope] = info;
      return info;
    }

    if (auto subslice = memdesc.getDefiningOp<ttng::TMEMSubSliceOp>()) {
      auto baseInfo = getOrCreate(subslice.getSrc(), rewriter, scope);
      if (!baseInfo)
        return std::nullopt;

      auto baseTy = cast<ttg::MemDescType>(subslice.getSrc().getType());
      if (baseTy.getRank() < 2 || memTy.getRank() != 2)
        return std::nullopt;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(subslice);
      auto loc = subslice.getLoc();
      int64_t stride = baseTy.getShape().front();
      if (baseTy.getRank() > 2)
        stride = product(baseTy.getShape().drop_front(1));
      int64_t offset = subslice.getN();
      auto offsetVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(offset));
      auto strideVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(stride));
      auto offsetEls = arith::MulIOp::create(
          rewriter, loc, rewriter.getI32Type(), offsetVal, strideVal);
      Value ptr = tt::AddPtrOp::create(rewriter, loc, baseInfo->ptr.getType(),
                                       baseInfo->ptr, offsetEls);
      ptr = remapToScope(ptr, rewriter, scope, loc);
      auto layout = getScratchEncoding(rewriter, memdesc, memTy);
      auto tensorTy = RankedTensorType::get(memTy.getShape(),
                                            memTy.getElementType(), layout);

      ScratchInfo info{ptr, tensorTy};
      return info;
    }

    if (auto view = memdesc.getDefiningOp<ttg::MemDescIndexOp>()) {
      auto baseInfo = getOrCreate(view.getSrc(), rewriter, scope);
      if (!baseInfo)
        return std::nullopt;

      auto baseTy = cast<ttg::MemDescType>(view.getSrc().getType());
      if (baseTy.getRank() < 2)
        return std::nullopt;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(view);
      auto loc = view.getLoc();
      Value idx = view.getIndex();
      idx = castToI32(rewriter, loc, idx);
      int64_t stride = product(baseTy.getShape().drop_front(1));
      auto strideVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(stride));
      auto offset = arith::MulIOp::create(rewriter, loc, rewriter.getI32Type(),
                                          idx, strideVal);
      Value ptr = tt::AddPtrOp::create(rewriter, loc, baseInfo->ptr.getType(),
                                       baseInfo->ptr, offset);
      ptr = remapToScope(ptr, rewriter, scope, loc);
      auto layout = getScratchEncoding(rewriter, memdesc, memTy);
      auto tensorTy = RankedTensorType::get(memTy.getShape(),
                                            memTy.getElementType(), layout);

      ScratchInfo info{ptr, tensorTy};
      return info;
    }

    if (auto view = memdesc.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      auto baseInfo = getOrCreate(view.getSrc(), rewriter, scope);
      if (!baseInfo)
        return std::nullopt;

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(view);
      auto loc = view.getLoc();
      Value ptr = baseInfo->ptr;
      auto ptrTy = triton::getPointerType(memTy.getElementType());
      if (ptr.getType() != ptrTy) {
        ptr = tt::BitcastOp::create(rewriter, loc, ptrTy, ptr);
      }
      ptr = remapToScope(ptr, rewriter, scope, loc);

      auto layout = getScratchEncoding(rewriter, memdesc, memTy);
      auto tensorTy = RankedTensorType::get(memTy.getShape(),
                                            memTy.getElementType(), layout);

      ScratchInfo info{ptr, tensorTy};
      return info;
    }

    return std::nullopt;
  }

private:
  Value remapToScope(Value value, PatternRewriter &rewriter, Region *scope,
                     Location loc) {
    if (!scope || isValueAvailableInScope(value, scope))
      return value;

    auto *parentOp = scope->getParentOp();
    auto partitions = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
        parentOp ? parentOp : nullptr);
    if (!partitions)
      return value;

    unsigned captureIdx = partitions.getNumOperands();
    for (auto [i, capture] :
         llvm::enumerate(partitions.getExplicitCaptures())) {
      if (capture == value) {
        captureIdx = i;
        break;
      }
    }

    if (captureIdx == partitions.getNumOperands()) {
      partitions->insertOperands(captureIdx, value);
      for (Region &region : partitions.getPartitionRegions()) {
        region.addArgument(value.getType(), loc);
      }
    }

    return scope->getArgument(captureIdx);
  }

  DenseMap<Value, DenseMap<Region *, ScratchInfo>> scratchMap;
};

Value createScratchAndStore(PatternRewriter &rewriter, Location loc, Value val,
                            RankedTensorType tensorTy) {
  int64_t elSize = tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t alignment = std::max<int64_t>(elSize, 16);
  int64_t sizeInBytes = product(tensorTy.getShape()) * elSize;
  auto ptrTy = triton::getPointerType(tensorTy.getElementType());
  auto allocOp = createThirdPartyScratchAlloc(rewriter, loc, ptrTy, sizeInBytes,
                                              alignment);
  allocOp->setDiscardableAttr("tt.divisibility",
                              rewriter.getI64IntegerAttr(alignment));
  createStoreScratchMemory(rewriter, loc, allocOp.getResult(), val, tensorTy);
  return allocOp.getResult();
}

Region *getScratchScopeRegion(Operation *anchor) {
  Region *region = anchor->getParentRegion();
  while (region) {
    if (auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(region->getParentOp())) {
      if (region == &wsOp.getDefaultRegion()) {
        region = wsOp->getParentRegion();
        continue;
      }
      return region;
    }
    if (isa<ttg::WarpSpecializePartitionsOp>(region->getParentOp()))
      return region;
    if (isa<tt::FuncOp>(region->getParentOp()))
      return region;
    region = region->getParentRegion();
  }
  llvm::report_fatal_error("getScratchScopeRegion called on an op that is not "
                           "contained in a function");
}

// ------------------------------------------------------------
// Utility functions
// ------------------------------------------------------------

Type getElementType(Type ty) {
  if (auto shaped = dyn_cast<ShapedType>(ty))
    return shaped.getElementType();
  return ty;
}

bool isFloatLike(Type ty) { return isa<FloatType>(getElementType(ty)); }

Type getIntTypeLike(Type ty) {
  auto elem = dyn_cast<FloatType>(getElementType(ty));
  if (!elem)
    return Type();

  auto *ctx = ty.getContext();
  auto intElem = IntegerType::get(ctx, elem.getWidth());
  if (auto ranked = dyn_cast<RankedTensorType>(ty)) {
    return RankedTensorType::get(ranked.getShape(), intElem,
                                 ranked.getEncoding());
  }
  if (isa<FloatType>(ty))
    return intElem;
  llvm::report_fatal_error("expected FloatType or RankedTensorType");
}

unsigned getIntBitwidth(Type ty) {
  auto elem = cast<IntegerType>(getElementType(ty));
  return elem.getWidth();
}

Type getTypeWithElement(Type ty, Type elemTy) {
  return cast<RankedTensorType>(ty).clone(elemTy);
}

Value getIntConstantLike(PatternRewriter &rewriter, Location loc, Type targetTy,
                         int64_t value) {
  if (auto shaped = dyn_cast<ShapedType>(targetTy)) {
    auto elem = cast<IntegerType>(shaped.getElementType());
    auto attr =
        DenseElementsAttr::get(shaped, rewriter.getIntegerAttr(elem, value));
    return arith::ConstantOp::create(rewriter, loc, attr);
  }
  auto intTy = cast<IntegerType>(targetTy);
  return arith::ConstantOp::create(rewriter, loc,
                                   rewriter.getIntegerAttr(intTy, value));
}

Value castIntValueToType(PatternRewriter &rewriter, Location loc, Value v,
                         Type targetTy) {
  if (v.getType() == targetTy)
    return v;

  unsigned srcWidth = getIntBitwidth(v.getType());
  unsigned dstWidth = getIntBitwidth(targetTy);
  if (dstWidth > srcWidth) {
    return arith::ExtUIOp::create(rewriter, loc, targetTy, v);
  }
  if (srcWidth > dstWidth) {
    return arith::TruncIOp::create(rewriter, loc, targetTy, v);
  }
  return v;
}

Value bitcastToInt(PatternRewriter &rewriter, Location loc, Value v) {
  if (isa<IntegerType>(getElementType(v.getType())))
    return v;
  auto intTy = getIntTypeLike(v.getType());
  return tt::BitcastOp::create(rewriter, loc, intTy, v);
}

Value bitcastToFloat(PatternRewriter &rewriter, Location loc, Value v,
                     Type floatTy) {
  return tt::BitcastOp::create(rewriter, loc, floatTy, v);
}

uint64_t stableStringHash(StringRef str) {
  uint64_t h = 14695981039346656037ull;
  for (uint8_t c : str.bytes()) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

uint64_t murmur64Mixer(uint64_t h) {
  h ^= h >> 33;
  h *= 0xff51afd7ed558ccd;
  h ^= h >> 33;
  h *= 0xc4ceb9fe1a85ec53;
  h ^= h >> 33;
  return h;
}

Value fpsanUnaryTagged(PatternRewriter &rewriter, Location loc, Value input,
                       UnaryOpId opId) {
  auto inI = bitcastToInt(rewriter, loc, input);
  uint64_t opIdHash = murmur64Mixer(getUnaryOpId(opId));
  auto opIdVal = getIntConstantLike(rewriter, loc, inI.getType(),
                                    static_cast<int64_t>(opIdHash));
  auto outI = arith::XOrIOp::create(rewriter, loc, inI, opIdVal);
  return bitcastToFloat(rewriter, loc, outI, input.getType());
}

Value fpsanFDiv(PatternRewriter &rewriter, Location loc, Value num, Value den) {
  auto numI = bitcastToInt(rewriter, loc, num);
  auto inv = bitcastToInt(
      rewriter, loc, fpsanUnaryTagged(rewriter, loc, den, UnaryOpId::DivInv));
  auto resI = arith::MulIOp::create(rewriter, loc, numI, inv);
  return bitcastToFloat(rewriter, loc, resI, num.getType());
}

Value fpsanSRem(PatternRewriter &rewriter, Location loc, Value num, Value den) {
  auto numI = bitcastToInt(rewriter, loc, num);
  auto denI = bitcastToInt(rewriter, loc, den);
  auto one = getIntConstantLike(rewriter, loc, denI.getType(), 1);
  auto denSafe = arith::OrIOp::create(rewriter, loc, denI, one);
  auto resI = arith::RemSIOp::create(rewriter, loc, numI, denSafe);
  return bitcastToFloat(rewriter, loc, resI, num.getType());
}

bool isIntLike(Type ty) { return isa<IntegerType>(getElementType(ty)); }

bool isNumericLike(Type ty) {
  Type elemTy = getElementType(ty);
  return isa<FloatType>(elemTy) || isa<IntegerType>(elemTy);
}

bool externHasNumericOperands(tt::ExternElementwiseOp op) {
  return llvm::all_of(op.getOperands(), [](Value operand) {
    return isNumericLike(operand.getType());
  });
}

bool externInvolvesFloatLike(tt::ExternElementwiseOp op) {
  return isFloatLike(op.getType()) ||
         llvm::any_of(op.getOperands(), [](Value operand) {
           return isFloatLike(operand.getType());
         });
}

Value castExternOperandToResultInt(PatternRewriter &rewriter, Location loc,
                                   Value operand, Type resultIntTy) {
  if (isFloatLike(operand.getType())) {
    return castIntValueToType(
        rewriter, loc, bitcastToInt(rewriter, loc, operand), resultIntTy);
  }
  if (isIntLike(operand.getType())) {
    return castIntValueToType(rewriter, loc, operand, resultIntTy);
  }
  return Value();
}

Value rotateLeftIntByAmount(PatternRewriter &rewriter, Location loc,
                            Value value, unsigned amount) {
  unsigned bitWidth = getIntBitwidth(value.getType());
  if (bitWidth == 0)
    return value;
  amount %= bitWidth;
  if (amount == 0)
    return value;

  auto leftShift = getIntConstantLike(rewriter, loc, value.getType(),
                                      static_cast<int64_t>(amount));
  auto rightShift = getIntConstantLike(rewriter, loc, value.getType(),
                                       static_cast<int64_t>(bitWidth - amount));
  auto left = arith::ShLIOp::create(rewriter, loc, value, leftShift);
  auto right = arith::ShRUIOp::create(rewriter, loc, value, rightShift);
  return arith::OrIOp::create(rewriter, loc, left, right);
}

Value fpsanVariadicExternTagged(PatternRewriter &rewriter, Location loc,
                                tt::ExternElementwiseOp op, uint64_t hash) {
  Type resultTy = op.getType();
  Type resultIntTy = getIntTypeLike(resultTy);

  Value sumI = getIntConstantLike(rewriter, loc, resultIntTy, 0);
  for (auto [argIdx, operand] : llvm::enumerate(op.getOperands())) {
    Value operandI =
        castExternOperandToResultInt(rewriter, loc, operand, resultIntTy);
    if (!operandI)
      return Value();
    Value rotated = rotateLeftIntByAmount(rewriter, loc, operandI,
                                          static_cast<unsigned>(argIdx));
    sumI = arith::AddIOp::create(rewriter, loc, sumI, rotated);
  }

  auto hashVal = getIntConstantLike(rewriter, loc, resultIntTy,
                                    static_cast<int64_t>(hash));
  auto outI = arith::XOrIOp::create(rewriter, loc, sumI, hashVal);
  return bitcastToFloat(rewriter, loc, outI, resultTy);
}

std::optional<ScratchInfo>
createOperandScratch(PatternRewriter &rewriter, Location loc,
                     TmemScratchManager &scratch, Value memdesc,
                     ttg::MemDescType memTy, bool isTmem, Region *scope) {
  auto layout = scratch.getScratchEncoding(rewriter, memdesc, memTy);
  auto tensorTy =
      RankedTensorType::get(memTy.getShape(), memTy.getElementType(), layout);
  Value fullVal;
  if (isTmem) {
    auto info = scratch.getOrCreate(memdesc, rewriter, scope);
    if (!info)
      return std::nullopt;
    fullVal = createLoadScratchMemory(rewriter, loc, info->ptr, tensorTy);
    if (!fullVal)
      return std::nullopt;
  } else {
    fullVal =
        ttg::LocalLoadOp::create(rewriter, loc, tensorTy, memdesc, Value())
            .getResult();
  }
  int64_t elSize = memTy.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t alignment = std::max<int64_t>(elSize, 16);
  int64_t sizeInBytes = product(memTy.getShape()) * elSize;
  auto ptrTy = triton::getPointerType(memTy.getElementType());
  auto allocOp = createThirdPartyScratchAlloc(rewriter, loc, ptrTy, sizeInBytes,
                                              alignment);
  allocOp->setDiscardableAttr("tt.divisibility",
                              rewriter.getI64IntegerAttr(alignment));
  Value ptr = allocOp.getResult();
  if (!createStoreScratchMemory(rewriter, loc, ptr, fullVal, tensorTy))
    return std::nullopt;
  return ScratchInfo{ptr, tensorTy};
}

Value createAsyncToken(PatternRewriter &rewriter, Location loc,
                       ValueRange deps) {
  return ttg::AsyncCommitGroupOp::create(rewriter, loc, deps).getResult();
}

Value expandAllSlicedDims(PatternRewriter &rewriter, Location loc,
                          Value tensor) {
  auto type = cast<RankedTensorType>(tensor.getType());
  auto sliceEncoding = dyn_cast<ttg::SliceEncodingAttr>(type.getEncoding());
  while (sliceEncoding) {
    tensor = expandOuterSlicedDim(rewriter, loc, tensor);
    type = cast<RankedTensorType>(tensor.getType());
    sliceEncoding = dyn_cast<ttg::SliceEncodingAttr>(type.getEncoding());
  }
  return tensor;
}

Value createPointerTensorStrided2D(PatternRewriter &rewriter, Location loc,
                                   Value base, RankedTensorType resultTy,
                                   int64_t stride1) {
  auto shape = resultTy.getShape();
  auto encoding = cast<ttg::DistributedEncodingTrait>(resultTy.getEncoding());
  auto ptrTy = base.getType();
  auto ptrTensorTy = RankedTensorType::get(shape, ptrTy, encoding);
  Value ptrTensor = tt::SplatOp::create(rewriter, loc, ptrTensorTy, base);
  auto i32Ty = rewriter.getI32Type();
  auto offsetsTy = RankedTensorType::get(shape, i32Ty, encoding);

  auto dim0Enc = getSingleDimSliceEncoding(encoding, 0);
  auto dim0Ty = RankedTensorType::get({shape[0]}, i32Ty, dim0Enc);
  auto range0 = tt::MakeRangeOp::create(rewriter, loc, dim0Ty, 0, shape[0]);
  auto stride0 = createConstIntTensor(rewriter, loc, 1, dim0Ty);
  auto off0 = arith::MulIOp::create(rewriter, loc, dim0Ty, range0, stride0);
  auto off0Exp = expandAllSlicedDims(rewriter, loc, off0);
  if (cast<RankedTensorType>(off0Exp.getType()).getShape() != shape) {
    off0Exp = tt::BroadcastOp::create(rewriter, loc, offsetsTy, off0Exp);
  }
  ptrTensor =
      tt::AddPtrOp::create(rewriter, loc, ptrTensorTy, ptrTensor, off0Exp);

  auto dim1Enc = getSingleDimSliceEncoding(encoding, 1);
  auto dim1Ty = RankedTensorType::get({shape[1]}, i32Ty, dim1Enc);
  auto range1 = tt::MakeRangeOp::create(rewriter, loc, dim1Ty, 0, shape[1]);
  auto stride1Const = createConstIntTensor(rewriter, loc, stride1, dim1Ty);
  auto off1 =
      arith::MulIOp::create(rewriter, loc, dim1Ty, range1, stride1Const);
  auto off1Exp = expandAllSlicedDims(rewriter, loc, off1);
  if (cast<RankedTensorType>(off1Exp.getType()).getShape() != shape) {
    off1Exp = tt::BroadcastOp::create(rewriter, loc, offsetsTy, off1Exp);
  }
  ptrTensor =
      tt::AddPtrOp::create(rewriter, loc, ptrTensorTy, ptrTensor, off1Exp);

  return ptrTensor;
}

Value loadScratchStrided2D(PatternRewriter &rewriter, Location loc, Value base,
                           RankedTensorType tensorTy, int64_t stride1) {
  auto ptrTensor =
      createPointerTensorStrided2D(rewriter, loc, base, tensorTy, stride1);
  return tt::LoadOp::create(rewriter, loc, ptrTensor, CacheModifier::NONE,
                            EvictionPolicy::NORMAL, false);
}

Operation *storeScratchStrided2D(PatternRewriter &rewriter, Location loc,
                                 Value base, Value tensor,
                                 RankedTensorType tensorTy, int64_t stride1) {
  auto ptrTensor =
      createPointerTensorStrided2D(rewriter, loc, base, tensorTy, stride1);
  return tt::StoreOp::create(rewriter, loc, ptrTensor, tensor,
                             CacheModifier::NONE, EvictionPolicy::NORMAL);
}

Value unpackPackedFp4Slice(PatternRewriter &rewriter, Location loc,
                           Value packedSlice, Value kI32) {
  Value packedI = bitcastToInt(rewriter, loc, packedSlice);
  auto intTy = packedI.getType();

  Value one =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
  Value four =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(4));
  Value isOdd = arith::AndIOp::create(rewriter, loc, kI32, one);
  Value shiftI32 = arith::MulIOp::create(rewriter, loc, isOdd, four);
  Value shiftI8 =
      arith::TruncIOp::create(rewriter, loc, rewriter.getI8Type(), shiftI32);
  Value shiftTensor = tt::SplatOp::create(rewriter, loc, intTy, shiftI8);

  Value shifted = arith::ShRUIOp::create(rewriter, loc, packedI, shiftTensor);
  Value mask = getIntConstantLike(rewriter, loc, intTy, 0x0F);
  return arith::AndIOp::create(rewriter, loc, shifted, mask);
}

struct DotScaleConfig {
  Value aScalePtr;
  Value bScalePtr;
  RankedTensorType aScaleTileTy;
  RankedTensorType bScaleTileTy;
  int64_t aScaleStride = 0;
  int64_t bScaleStride = 0;
  int64_t aKPackFactor = 1;
  int64_t bKPackFactor = 1;
  int64_t aScaleFactor = 0;
  int64_t bScaleFactor = 0;
};

Value loadScaleSlice(PatternRewriter &rewriter, Location loc, bool isLhs,
                     const DotScaleConfig &scale, Value tileIdx, Value kI32) {
  auto i32Ty = rewriter.getI32Type();
  Value ptr = isLhs ? scale.aScalePtr : scale.bScalePtr;
  int64_t sFactor = isLhs ? scale.aScaleFactor : scale.bScaleFactor;
  int64_t sStride = isLhs ? scale.aScaleStride : scale.bScaleStride;
  int64_t loadStride = isLhs ? scale.aScaleStride : 1;
  auto tileTy = isLhs ? scale.aScaleTileTy : scale.bScaleTileTy;
  Value tilePtr =
      tt::AddPtrOp::create(rewriter, loc, ptr.getType(), ptr, tileIdx);
  Value sFactorConst = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(sFactor));
  Value kGrp = arith::DivUIOp::create(rewriter, loc, kI32, sFactorConst);
  Value sStrideConst = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(sStride));
  Value offset = arith::MulIOp::create(rewriter, loc, kGrp, sStrideConst);
  Value slicePtr =
      tt::AddPtrOp::create(rewriter, loc, ptr.getType(), tilePtr, offset);
  return loadScratchStrided2D(rewriter, loc, slicePtr, tileTy, loadStride);
}

Value emulateDotStep(PatternRewriter &rewriter, Location loc, Value aSlice,
                     Value bSlice, Value aScaleSlice, Value bScaleSlice,
                     int64_t m, int64_t n,
                     ttg::DistributedEncodingTrait accLayout,
                     IntegerType accElem) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto fullTy = RankedTensorType::get({m, n}, accElem, accLayout);
  auto aI = bitcastToInt(rewriter, loc, aSlice);
  auto bI = bitcastToInt(rewriter, loc, bSlice);
  auto scaleElem = rewriter.getI16Type();
  if (aScaleSlice) {
    auto aScaleI = bitcastToInt(rewriter, loc, aScaleSlice);
    aI = castIntValueToType(rewriter, loc, aI,
                            getTypeWithElement(aI.getType(), scaleElem));
    aScaleI =
        castIntValueToType(rewriter, loc, aScaleI,
                           getTypeWithElement(aScaleI.getType(), scaleElem));
    aI = arith::MulIOp::create(rewriter, loc, aI, aScaleI);
  }
  if (bScaleSlice) {
    auto bScaleI = bitcastToInt(rewriter, loc, bScaleSlice);
    bI = castIntValueToType(rewriter, loc, bI,
                            getTypeWithElement(bI.getType(), scaleElem));
    bScaleI =
        castIntValueToType(rewriter, loc, bScaleI,
                           getTypeWithElement(bScaleI.getType(), scaleElem));
    bI = arith::MulIOp::create(rewriter, loc, bI, bScaleI);
  }
  aI = castIntValueToType(rewriter, loc, aI,
                          getTypeWithElement(aI.getType(), accElem));
  bI = castIntValueToType(rewriter, loc, bI,
                          getTypeWithElement(bI.getType(), accElem));
  Value aFull = tt::BroadcastOp::create(rewriter, loc, fullTy, aI);
  Value bFull = tt::BroadcastOp::create(rewriter, loc, fullTy, bI);
  return arith::MulIOp::create(rewriter, loc, aFull, bFull);
}

std::optional<scf::ForOp> emitMmaEmulationLoops(
    PatternRewriter &rewriter, Location loc, Value aPtr, Value bPtr, Value dPtr,
    int64_t m, int64_t n, int64_t k, int64_t tileM, int64_t tileN,
    RankedTensorType aTileTy, RankedTensorType bTileTy,
    RankedTensorType accTileTy, ttg::DistributedEncodingTrait accLayout,
    IntegerType accElem, Value useDInt, Value predInt, int64_t aStride,
    int64_t bStride, int64_t dStride, const DotScaleConfig &scale = {}) {
  if ((m % tileM) != 0 || (n % tileN) != 0)
    return std::nullopt;

  OpBuilder::InsertionGuard guard(rewriter);
  Value zero =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  Value mUpper =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(m));
  Value nUpper =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(n));
  Value mStep = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getI32IntegerAttr(tileM));
  Value nStep = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getI32IntegerAttr(tileN));

  auto mLoop = scf::ForOp::create(rewriter, loc, zero, mUpper, mStep);
  rewriter.setInsertionPointToStart(mLoop.getBody());
  Value mIdx = mLoop.getInductionVar();
  auto nLoop = scf::ForOp::create(rewriter, loc, zero, nUpper, nStep);
  rewriter.setInsertionPointToStart(nLoop.getBody());
  Value nIdx = nLoop.getInductionVar();

  auto i32Ty = rewriter.getI32Type();
  Value mIdxI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, mIdx);
  Value nIdxI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, nIdx);
  Value mConst =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(m));
  Value bStrideConst = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(bStride));

  Value nMulM = arith::MulIOp::create(rewriter, loc, nIdxI32, mConst);
  Value dOffset = arith::AddIOp::create(rewriter, loc, mIdxI32, nMulM);
  Value dTilePtr =
      tt::AddPtrOp::create(rewriter, loc, dPtr.getType(), dPtr, dOffset);
  Value accTile =
      loadScratchStrided2D(rewriter, loc, dTilePtr, accTileTy, dStride);
  Value accTileI = bitcastToInt(rewriter, loc, accTile);

  Value aTilePtr =
      tt::AddPtrOp::create(rewriter, loc, aPtr.getType(), aPtr, mIdxI32);
  Value bOffset = arith::MulIOp::create(rewriter, loc, nIdxI32, bStrideConst);
  Value bTilePtr =
      tt::AddPtrOp::create(rewriter, loc, bPtr.getType(), bPtr, bOffset);

  auto aSliceTy =
      RankedTensorType::get({tileM, 1}, aTileTy.getElementType(), accLayout);
  auto bSliceTy =
      RankedTensorType::get({1, tileN}, bTileTy.getElementType(), accLayout);
  Value aStrideVal = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(aStride));

  Value zeroSum = getIntConstantLike(rewriter, loc, accTileI.getType(), 0);
  Value kUpper =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(k));
  Value kStep =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
  auto kLoop = scf::ForOp::create(rewriter, loc, zero, kUpper, kStep, zeroSum);
  rewriter.setInsertionPointToStart(kLoop.getBody());
  Value kIdx = kLoop.getInductionVar();
  Value kI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, kIdx);
  Value aKIdx = kI32;
  Value bKIdx = kI32;
  if (scale.aKPackFactor == 2) {
    Value aPackFactor = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(scale.aKPackFactor));
    aKIdx = arith::DivUIOp::create(rewriter, loc, kI32, aPackFactor);
  }
  if (scale.bKPackFactor == 2) {
    Value bPackFactor = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(scale.bKPackFactor));
    bKIdx = arith::DivUIOp::create(rewriter, loc, kI32, bPackFactor);
  }
  Value aOffset =
      arith::MulIOp::create(rewriter, loc, i32Ty, aKIdx, aStrideVal);
  Value aSlicePtr =
      tt::AddPtrOp::create(rewriter, loc, aPtr.getType(), aTilePtr, aOffset);
  Value aSlice =
      loadScratchStrided2D(rewriter, loc, aSlicePtr, aSliceTy, aStride);
  Value bSlicePtr =
      tt::AddPtrOp::create(rewriter, loc, bPtr.getType(), bTilePtr, bKIdx);
  Value bSlice =
      loadScratchStrided2D(rewriter, loc, bSlicePtr, bSliceTy, bStride);
  Value aScaleSlice;
  if (scale.aScalePtr) {
    if (scale.aKPackFactor == 2)
      aSlice = unpackPackedFp4Slice(rewriter, loc, aSlice, kI32);
    aScaleSlice =
        loadScaleSlice(rewriter, loc, /*isLhs=*/true, scale, mIdxI32, kI32);
  }
  Value bScaleSlice;
  if (scale.bScalePtr) {
    if (scale.bKPackFactor == 2)
      bSlice = unpackPackedFp4Slice(rewriter, loc, bSlice, kI32);
    bScaleSlice =
        loadScaleSlice(rewriter, loc, /*isLhs=*/false, scale, nIdxI32, kI32);
  }
  Value partial = emulateDotStep(rewriter, loc, aSlice, bSlice, aScaleSlice,
                                 bScaleSlice, tileM, tileN, accLayout, accElem);
  Value acc = kLoop.getRegionIterArgs()[0];
  Value next = arith::AddIOp::create(rewriter, loc, acc, partial);
  scf::YieldOp::create(rewriter, loc, next);
  rewriter.setInsertionPointAfter(kLoop);
  Value sum = kLoop.getResult(0);

  Value useDMask =
      tt::SplatOp::create(rewriter, loc, accTileI.getType(), useDInt);
  Value accInitI = arith::MulIOp::create(rewriter, loc, accTileI, useDMask);
  Value outI = arith::AddIOp::create(rewriter, loc, sum, accInitI);

  Value predMask =
      tt::SplatOp::create(rewriter, loc, accTileI.getType(), predInt);
  Value oneI = getIntConstantLike(rewriter, loc, accTileI.getType(), 1);
  Value predInv = arith::SubIOp::create(rewriter, loc, oneI, predMask);
  Value outMasked = arith::MulIOp::create(rewriter, loc, outI, predMask);
  Value accMasked = arith::MulIOp::create(rewriter, loc, accTileI, predInv);
  Value outSelI = arith::AddIOp::create(rewriter, loc, outMasked, accMasked);
  Value out = bitcastToFloat(rewriter, loc, outSelI, accTileTy);
  storeScratchStrided2D(rewriter, loc, dTilePtr, out, accTileTy, dStride);

  return mLoop;
}

//----------------------------------------
// Patterns
//----------------------------------------

template <typename OpF, typename OpI>
struct BinaryFloatToIntPattern : public OpRewritePattern<OpF> {
  using OpRewritePattern<OpF>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpF op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto lhsI = bitcastToInt(rewriter, loc, op.getLhs());
    auto rhsI = bitcastToInt(rewriter, loc, op.getRhs());
    auto resI = OpI::create(rewriter, loc, lhsI, rhsI);
    auto resF = bitcastToFloat(rewriter, loc, resI, op.getType());
    rewriter.replaceOp(op, resF);
    return success();
  }
};

struct DivFOpPattern : public OpRewritePattern<arith::DivFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::DivFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(
        op, fpsanFDiv(rewriter, op.getLoc(), op.getLhs(), op.getRhs()));
    return success();
  }
};

struct PreciseDivFOpPattern : public OpRewritePattern<tt::PreciseDivFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tt::PreciseDivFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(op,
                       fpsanFDiv(rewriter, op.getLoc(), op.getX(), op.getY()));
    return success();
  }
};

struct RemFOpPattern : public OpRewritePattern<arith::RemFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::RemFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(
        op, fpsanSRem(rewriter, op.getLoc(), op.getLhs(), op.getRhs()));
    return success();
  }
};

struct FmaPattern : public OpRewritePattern<math::FmaOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::FmaOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto aI = bitcastToInt(rewriter, loc, op.getA());
    auto bI = bitcastToInt(rewriter, loc, op.getB());
    auto cI = bitcastToInt(rewriter, loc, op.getC());
    auto mul = arith::MulIOp::create(rewriter, loc, aI, bI);
    auto sum = arith::AddIOp::create(rewriter, loc, mul, cI);
    auto resF = bitcastToFloat(rewriter, loc, sum, op.getType());
    rewriter.replaceOp(op, resF);
    return success();
  }
};

struct ExtFOpPattern : public OpRewritePattern<arith::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::ExtFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto inI = bitcastToInt(rewriter, loc, op.getIn());
    auto outI =
        castIntValueToType(rewriter, loc, inI, getIntTypeLike(op.getType()));
    auto outF = bitcastToFloat(rewriter, loc, outI, op.getType());
    rewriter.replaceOp(op, outF);
    return success();
  }
};

struct TruncFOpPattern : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto inI = bitcastToInt(rewriter, loc, op.getIn());
    auto outI =
        castIntValueToType(rewriter, loc, inI, getIntTypeLike(op.getType()));
    auto outF = bitcastToFloat(rewriter, loc, outI, op.getType());
    rewriter.replaceOp(op, outF);
    return success();
  }
};

struct FpToFpPattern : public OpRewritePattern<tt::FpToFpOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tt::FpToFpOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto inI = bitcastToInt(rewriter, loc, op.getSrc());
    auto outI =
        castIntValueToType(rewriter, loc, inI, getIntTypeLike(op.getType()));
    auto outF = bitcastToFloat(rewriter, loc, outI, op.getType());
    rewriter.replaceOp(op, outF);
    return success();
  }
};

struct Fp4ToFpPattern : public OpRewritePattern<ttg::Fp4ToFpOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttg::Fp4ToFpOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    if (!srcTy || !dstTy)
      return failure();
    auto srcElemTy = dyn_cast<IntegerType>(srcTy.getElementType());
    if (!srcElemTy || srcElemTy.getWidth() != 8)
      return failure();

    int64_t axis = op.getAxis();
    int64_t rank = srcTy.getRank();
    auto dstIntTy = cast<RankedTensorType>(getIntTypeLike(dstTy));
    auto halfIntTy = srcTy.clone(dstIntTy.getElementType());

    auto loc = op.getLoc();
    auto mask = getIntConstantLike(rewriter, loc, srcTy, 0x0F);
    auto four = getIntConstantLike(rewriter, loc, srcTy, 4);
    Value lo = arith::AndIOp::create(rewriter, loc, op.getSrc(), mask);
    Value hi = arith::ShRUIOp::create(rewriter, loc, op.getSrc(), four);
    auto loI = castIntValueToType(rewriter, loc, lo, halfIntTy);
    auto hiI = castIntValueToType(rewriter, loc, hi, halfIntTy);
    Value joined = tt::JoinOp::create(rewriter, loc, loI, hiI);

    auto order = llvm::to_vector(llvm::seq<int32_t>(axis + 1));
    order.push_back(rank);
    llvm::append_range(order, llvm::seq<int32_t>(axis + 1, rank));
    auto transposed = tt::TransOp::create(rewriter, loc, joined, order);

    Value result =
        tt::ReshapeOp::create(rewriter, loc, dstTy.getShape(), transposed);
    if (result.getType() != dstIntTy)
      result = ttg::ConvertLayoutOp::create(rewriter, loc, dstIntTy, result);

    rewriter.replaceOp(op, bitcastToFloat(rewriter, loc, result, dstTy));
    return success();
  }
};

struct DotPattern : public OpRewritePattern<tt::DotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tt::DotOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
    auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
    auto cTy = dyn_cast<RankedTensorType>(op.getC().getType());
    if (!aTy || !bTy || !cTy)
      return failure();
    if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2)
      return failure();
    if (!aTy.getEncoding() || !bTy.getEncoding() || !cTy.getEncoding())
      return failure();

    auto aShape = aTy.getShape();
    auto bShape = bTy.getShape();
    auto cShape = cTy.getShape();
    if (aShape[1] != bShape[0] || aShape[0] != cShape[0] ||
        bShape[1] != cShape[1])
      return failure();

    auto loc = op.getLoc();
    int64_t m = aShape[0];
    int64_t k = aShape[1];
    int64_t n = bShape[1];

    auto accElem = IntegerType::get(
        rewriter.getContext(), cTy.getElementType().getIntOrFloatBitWidth());
    Value useDInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));
    Value predInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));

    int64_t tileM = std::min<int64_t>(kTileM, m);
    int64_t tileN = std::min<int64_t>(kTileN, n);

    // Use optimized blocked layouts for emulation tiles instead of the
    // original dot encodings.  Encodings like AMDWmmaEncodingAttr impose
    // minimum shape requirements (e.g. >= 16x16) that the small emulation
    // tiles (kTileM x kTileN = 8x8) cannot satisfy.
    auto accLayout = getOptimizedBlockedEncoding(rewriter, {tileM, tileN},
                                                 cTy.getElementType());
    auto aLayout =
        getOptimizedBlockedEncoding(rewriter, {tileM, k}, aTy.getElementType());
    auto bLayout =
        getOptimizedBlockedEncoding(rewriter, {k, tileN}, bTy.getElementType());

    auto accTileTy =
        RankedTensorType::get({tileM, tileN}, cTy.getElementType(), accLayout);
    auto aTileTy =
        RankedTensorType::get({tileM, k}, aTy.getElementType(), aLayout);
    auto bTileTy =
        RankedTensorType::get({k, tileN}, bTy.getElementType(), bLayout);

    Value aPtr = createScratchAndStore(rewriter, loc, op.getA(), aTy);
    Value bPtr = createScratchAndStore(rewriter, loc, op.getB(), bTy);
    Value dPtr = createScratchAndStore(rewriter, loc, op.getC(), cTy);

    // Each warp may only store a subset of each tile's rows, so a barrier is
    // needed to make all scratch stores visible before the loops read them.
    ttg::BarrierOp::create(rewriter, loc,
                           ttg::AddrSpace::GlobalRead |
                               ttg::AddrSpace::GlobalWrite);

    auto mLoop = emitMmaEmulationLoops(
        rewriter, loc, aPtr, bPtr, dPtr, m, n, k, tileM, tileN, aTileTy,
        bTileTy, accTileTy, accLayout, accElem, useDInt, predInt,
        /*aStride=*/m, /*bStride=*/k, /*dStride=*/m);
    if (!mLoop)
      return failure();
    rewriter.setInsertionPointAfter(*mLoop);

    // Same reason: each warp may only write a subset of D's rows in the loop,
    // so synchronize before the final load.
    ttg::BarrierOp::create(rewriter, loc,
                           ttg::AddrSpace::GlobalRead |
                               ttg::AddrSpace::GlobalWrite);

    Value out = loadScratchStrided2D(rewriter, loc, dPtr, cTy, /*stride1=*/m);
    if (!out)
      return failure();
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct DotScaledPattern : public OpRewritePattern<tt::DotScaledOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tt::DotScaledOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto aScale = op.getAScale();
    auto bScale = op.getBScale();
    auto aTy = dyn_cast<RankedTensorType>(op.getA().getType());
    auto bTy = dyn_cast<RankedTensorType>(op.getB().getType());
    auto cTy = dyn_cast<RankedTensorType>(op.getC().getType());
    auto aScaleTy = aScale ? dyn_cast<RankedTensorType>(aScale.getType())
                           : RankedTensorType();
    auto bScaleTy = bScale ? dyn_cast<RankedTensorType>(bScale.getType())
                           : RankedTensorType();
    if (!aTy || !bTy || !cTy || (aScale && !aScaleTy) || (bScale && !bScaleTy))
      return failure();
    if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2 ||
        (aScale && aScaleTy.getRank() != 2) ||
        (bScale && bScaleTy.getRank() != 2))
      return failure();
    if (!aTy.getEncoding() || !bTy.getEncoding() || !cTy.getEncoding() ||
        (aScale && !aScaleTy.getEncoding()) ||
        (bScale && !bScaleTy.getEncoding()))
      return failure();
    // TODO: Support M/N packing.
    if (!op.getLhsKPack() || !op.getRhsKPack())
      return failure();

    auto aShape = aTy.getShape();
    auto bShape = bTy.getShape();
    auto cShape = cTy.getShape();
    if (aShape[0] != cShape[0] || bShape[1] != cShape[1])
      return failure();

    int64_t aKPackFactor = 1;
    int64_t bKPackFactor = 1;
    if (op.getAElemType() == tt::ScaleDotElemType::E2M1)
      aKPackFactor = 2;
    if (op.getBElemType() == tt::ScaleDotElemType::E2M1)
      bKPackFactor = 2;
    int64_t aPackedK = aShape[1];
    int64_t bPackedK = bShape[0];
    int64_t k = aPackedK * aKPackFactor;
    if (k != bPackedK * bKPackFactor)
      return failure();

    auto loc = op.getLoc();
    int64_t m = cShape[0];
    int64_t n = cShape[1];
    if ((aScale && aScaleTy.getShape()[0] != m) ||
        (bScale && bScaleTy.getShape()[0] != n))
      return failure();

    auto accElem = IntegerType::get(
        rewriter.getContext(), cTy.getElementType().getIntOrFloatBitWidth());
    Value useDInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));
    Value predInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));

    int64_t tileM = std::min<int64_t>(kTileM, m);
    int64_t tileN = std::min<int64_t>(kTileN, n);

    auto accLayout = getOptimizedBlockedEncoding(rewriter, {tileM, tileN},
                                                 cTy.getElementType());
    auto aLayout = getOptimizedBlockedEncoding(rewriter, {tileM, aPackedK},
                                               aTy.getElementType());
    auto bLayout = getOptimizedBlockedEncoding(rewriter, {bPackedK, tileN},
                                               bTy.getElementType());

    auto accTileTy =
        RankedTensorType::get({tileM, tileN}, cTy.getElementType(), accLayout);
    auto aTileTy =
        RankedTensorType::get({tileM, aPackedK}, aTy.getElementType(), aLayout);
    auto bTileTy =
        RankedTensorType::get({bPackedK, tileN}, bTy.getElementType(), bLayout);

    auto aPtr = createScratchAndStore(rewriter, loc, op.getA(), aTy);
    auto bPtr = createScratchAndStore(rewriter, loc, op.getB(), bTy);
    auto dPtr = createScratchAndStore(rewriter, loc, op.getC(), cTy);

    auto aElemType = op.getAElemType();
    auto bElemType = op.getBElemType();
    bool skipAScale = aElemType == tt::ScaleDotElemType::BF16 ||
                      aElemType == tt::ScaleDotElemType::FP16;
    bool skipBScale = bElemType == tt::ScaleDotElemType::BF16 ||
                      bElemType == tt::ScaleDotElemType::FP16;

    DotScaleConfig scale;
    scale.aKPackFactor = aKPackFactor;
    scale.bKPackFactor = bKPackFactor;
    if (aScale && !skipAScale) {
      scale.aScalePtr = createScratchAndStore(rewriter, loc, aScale, aScaleTy);
      scale.aScaleStride = aScaleTy.getShape()[0];
      scale.aScaleFactor = op.deduceScaleFactor();
      scale.aScaleTileTy = RankedTensorType::get(
          {tileM, 1}, aScaleTy.getElementType(), accLayout);
    }
    if (bScale && !skipBScale) {
      scale.bScalePtr = createScratchAndStore(rewriter, loc, bScale, bScaleTy);
      scale.bScaleStride = bScaleTy.getShape()[0];
      scale.bScaleFactor = op.deduceScaleFactor();
      scale.bScaleTileTy = RankedTensorType::get(
          {1, tileN}, bScaleTy.getElementType(), accLayout);
    }

    ttg::BarrierOp::create(rewriter, loc,
                           ttg::AddrSpace::GlobalRead |
                               ttg::AddrSpace::GlobalWrite);

    auto mLoop = emitMmaEmulationLoops(
        rewriter, loc, aPtr, bPtr, dPtr, m, n, k, tileM, tileN, aTileTy,
        bTileTy, accTileTy, accLayout, accElem, useDInt, predInt,
        /*aStride=*/m, /*bStride=*/bPackedK, /*dStride=*/m, scale);
    if (!mLoop)
      return failure();
    rewriter.setInsertionPointAfter(*mLoop);

    ttg::BarrierOp::create(rewriter, loc,
                           ttg::AddrSpace::GlobalRead |
                               ttg::AddrSpace::GlobalWrite);

    Value out = loadScratchStrided2D(rewriter, loc, dPtr, cTy, /*stride1=*/m);
    if (!out)
      return failure();
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct TMEMLoadPattern : public OpRewritePattern<ttng::TMEMLoadOp> {
  TMEMLoadPattern(MLIRContext *ctx, TmemScratchManager *scratch)
      : OpRewritePattern(ctx), scratch(scratch) {}

  LogicalResult matchAndRewrite(ttng::TMEMLoadOp op,
                                PatternRewriter &rewriter) const override {
    Region *scope = getScratchScopeRegion(op);
    std::optional<ScratchInfo> info =
        scratch->getOrCreate(op.getSrc(), rewriter, scope);
    if (!info)
      return failure();

    Location loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    if (!resultTy.getEncoding())
      return failure();
    Value result = createLoadScratchMemory(rewriter, loc, info->ptr, resultTy);
    if (!result)
      return failure();

    if (op.getNumResults() == 1) {
      rewriter.replaceOp(op, result);
      return success();
    }
    SmallVector<Value> deps;
    if (op.getDep())
      deps.push_back(op.getDep());
    Value token = createAsyncToken(rewriter, loc, deps);
    rewriter.replaceOp(op, {result, token});
    return success();
  }

private:
  TmemScratchManager *scratch;
};

struct TMEMStorePattern : public OpRewritePattern<ttng::TMEMStoreOp> {
  TMEMStorePattern(MLIRContext *ctx, TmemScratchManager *scratch)
      : OpRewritePattern(ctx), scratch(scratch) {}

  LogicalResult matchAndRewrite(ttng::TMEMStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto scope = getScratchScopeRegion(op);
    auto info = scratch->getOrCreate(op.getDst(), rewriter, scope);
    if (!info)
      return failure();

    auto loc = op.getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTy.getEncoding())
      return failure();
    if (!createStoreScratchMemory(rewriter, loc, info->ptr, op.getSrc(), srcTy))
      return failure();

    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }
    SmallVector<Value> deps;
    if (op.getDep())
      deps.push_back(op.getDep());
    Value token = createAsyncToken(rewriter, loc, deps);
    rewriter.replaceOp(op, token);
    return success();
  }

private:
  TmemScratchManager *scratch;
};

struct TMEMCopyPattern : public OpRewritePattern<ttng::TMEMCopyOp> {
  TMEMCopyPattern(MLIRContext *ctx, TmemScratchManager *scratch)
      : OpRewritePattern(ctx), scratch(scratch) {}

  LogicalResult matchAndRewrite(ttng::TMEMCopyOp op,
                                PatternRewriter &rewriter) const override {
    auto scope = getScratchScopeRegion(op);
    auto info = scratch->getOrCreate(op.getDst(), rewriter, scope);
    if (!info)
      return failure();

    auto loc = op.getLoc();
    auto srcMemTy = cast<ttg::MemDescType>(op.getSrc().getType());
    auto srcRegTy =
        RankedTensorType::get(srcMemTy.getShape(), srcMemTy.getElementType(),
                              info->tensorType.getEncoding());
    Value srcReg =
        ttg::LocalLoadOp::create(rewriter, loc, srcRegTy, op.getSrc(), Value())
            .getResult();
    if (!createStoreScratchMemory(rewriter, loc, info->ptr, srcReg, srcRegTy))
      return failure();

    if (Value barrier = op.getBarrier()) {
      ttng::ArriveBarrierOp::create(rewriter, loc, barrier, 1, Value());
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  TmemScratchManager *scratch;
};

struct TCGen5CommitPattern : public OpRewritePattern<ttng::TCGen5CommitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ttng::TCGen5CommitOp op,
                                PatternRewriter &rewriter) const override {
    ttng::ArriveBarrierOp::create(rewriter, op.getLoc(), op.getBarrier(), 1,
                                  op.getPred());
    rewriter.eraseOp(op);
    return success();
  }
};

struct TCGen5MMAPattern : public OpRewritePattern<ttng::TCGen5MMAOp> {
  TCGen5MMAPattern(MLIRContext *ctx, TmemScratchManager *scratch)
      : OpRewritePattern(ctx), scratch(scratch) {}

  LogicalResult matchAndRewrite(ttng::TCGen5MMAOp op,
                                PatternRewriter &rewriter) const override {
    auto aMemTy = cast<ttg::MemDescType>(op.getA().getType());
    auto bMemTy = cast<ttg::MemDescType>(op.getB().getType());
    auto dMemTy = cast<ttg::MemDescType>(op.getD().getType());

    if (!isa<FloatType>(aMemTy.getElementType()) ||
        !isa<FloatType>(bMemTy.getElementType()) ||
        !isa<FloatType>(dMemTy.getElementType()))
      return failure();

    auto scope = getScratchScopeRegion(op);
    auto dInfo = scratch->getOrCreate(op.getD(), rewriter, scope);
    if (!dInfo)
      return failure();

    auto loc = op.getLoc();

    bool aIsTmem = isa<ttng::TensorMemorySpaceAttr>(aMemTy.getMemorySpace());
    bool bIsTmem = isa<ttng::TensorMemorySpaceAttr>(bMemTy.getMemorySpace());

    if ((aIsTmem && aMemTy.getRank() != 2) ||
        (bIsTmem && bMemTy.getRank() != 2) || dMemTy.getRank() != 2)
      return failure();

    auto aShape = aMemTy.getShape();
    auto bShape = bMemTy.getShape();
    if (aShape.size() != 2 || bShape.size() != 2)
      return failure();
    if (aShape[1] != bShape[0])
      return failure();
    int64_t m = aShape[0];
    int64_t k = aShape[1];
    int64_t n = bShape[1];

    auto *ctx = rewriter.getContext();
    auto accElem =
        IntegerType::get(ctx, dMemTy.getElementType().getIntOrFloatBitWidth());
    Value useDInt =
        arith::ExtUIOp::create(rewriter, loc, accElem, op.getUseD());
    Value predInt =
        arith::ExtUIOp::create(rewriter, loc, accElem, op.getPred());

    rewriter.setInsertionPoint(op);
    auto aScratch = createOperandScratch(rewriter, loc, *scratch, op.getA(),
                                         aMemTy, aIsTmem, scope);
    if (!aScratch)
      return failure();
    auto bScratch = createOperandScratch(rewriter, loc, *scratch, op.getB(),
                                         bMemTy, bIsTmem, scope);
    if (!bScratch)
      return failure();

    int64_t tileM = std::min<int64_t>(kTileM, m);
    int64_t tileN = std::min<int64_t>(kTileN, n);

    auto accTileLayout = getOptimizedBlockedEncoding(rewriter, {tileM, tileN},
                                                     dMemTy.getElementType());
    auto accTileTy = RankedTensorType::get(
        {tileM, tileN}, dMemTy.getElementType(), accTileLayout);
    auto aTileLayout = getOptimizedBlockedEncoding(rewriter, {tileM, k},
                                                   aMemTy.getElementType());
    auto aTileTy =
        RankedTensorType::get({tileM, k}, aMemTy.getElementType(), aTileLayout);
    auto bTileLayout = getOptimizedBlockedEncoding(rewriter, {k, tileN},
                                                   bMemTy.getElementType());
    auto bTileTy =
        RankedTensorType::get({k, tileN}, bMemTy.getElementType(), bTileLayout);

    auto mLoop = emitMmaEmulationLoops(
        rewriter, loc, aScratch->ptr, bScratch->ptr, dInfo->ptr, m, n, k, tileM,
        tileN, aTileTy, bTileTy, accTileTy, accTileLayout, accElem, useDInt,
        predInt, /*aStride=*/m, /*bStride=*/k, /*dStride=*/m);
    if (!mLoop)
      return failure();
    rewriter.setInsertionPointAfter(*mLoop);

    if (!op.getBarriers().empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(*mLoop);
      auto barriers = op.getBarriers();
      auto barrierPreds = op.getBarrierPreds();
      for (size_t i = 0; i < barriers.size(); ++i) {
        Value pred =
            arith::AndIOp::create(rewriter, loc, op.getPred(), barrierPreds[i]);
        ttng::ArriveBarrierOp::create(rewriter, loc, barriers[i], 1, pred);
      }
    }

    if (op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return success();
    }
    SmallVector<Value> deps;
    if (op.getAccDep())
      deps.push_back(op.getAccDep());
    Value token = createAsyncToken(rewriter, loc, deps);
    rewriter.replaceOp(op, token);
    return success();
  }

private:
  TmemScratchManager *scratch;
};

template <typename OpTy> struct UnaryPattern : public OpRewritePattern<OpTy> {
  UnaryPattern(MLIRContext *context, UnaryOpId unaryOpId)
      : OpRewritePattern<OpTy>(context), unaryOpId(unaryOpId) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(op, fpsanUnaryTagged(rewriter, op.getLoc(),
                                            op.getOperand(), unaryOpId));
    return success();
  }

private:
  UnaryOpId unaryOpId;
};

struct ExternElementwisePattern
    : public OpRewritePattern<tt::ExternElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getPure() || !isFloatLike(op.getType()) ||
        op.getNumOperands() == 0 || !externHasNumericOperands(op))
      return failure();

    uint64_t hash = stableStringHash(op.getSymbol());
    Value result = fpsanVariadicExternTagged(rewriter, op.getLoc(), op, hash);
    if (!result)
      return failure();
    rewriter.replaceOp(op, result);
    return success();
  }
};

class FpSanitizerPass
    : public impl::TritonInstrumentFpSanitizerBase<FpSanitizerPass> {
public:
  void runOnOperation() override {
    bool hasUnsupportedOperations = false;
    getOperation()->walk([&hasUnsupportedOperations](Operation *op) {
      if (isa<ttng::TCGen5MMAScaledOp>(op)) {
        hasUnsupportedOperations = true;
        llvm::errs() << "FpSanitizer error: Unsupported operation found: "
                     << op->getName() << "\n";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasUnsupportedOperations) {
      signalPassFailure();
    }

    TmemScratchManager scratch;
    RewritePatternSet patterns(&getContext());
    patterns.add<BinaryFloatToIntPattern<arith::AddFOp, arith::AddIOp>,
                 BinaryFloatToIntPattern<arith::SubFOp, arith::SubIOp>,
                 BinaryFloatToIntPattern<arith::MulFOp, arith::MulIOp>,
                 DivFOpPattern, PreciseDivFOpPattern, RemFOpPattern, FmaPattern,
                 ExtFOpPattern, TruncFOpPattern, FpToFpPattern, Fp4ToFpPattern,
                 DotPattern, DotScaledPattern>(&getContext());
    patterns.add<UnaryPattern<math::ExpOp>>(&getContext(), UnaryOpId::Exp);
    patterns.add<UnaryPattern<math::LogOp>>(&getContext(), UnaryOpId::Log);
    patterns.add<UnaryPattern<math::Exp2Op>>(&getContext(), UnaryOpId::Exp2);
    patterns.add<UnaryPattern<math::Log2Op>>(&getContext(), UnaryOpId::Log2);
    patterns.add<UnaryPattern<math::CosOp>>(&getContext(), UnaryOpId::Cos);
    patterns.add<UnaryPattern<math::SinOp>>(&getContext(), UnaryOpId::Sin);
    patterns.add<UnaryPattern<math::SqrtOp>>(&getContext(), UnaryOpId::Sqrt);
    patterns.add<UnaryPattern<math::RsqrtOp>>(&getContext(), UnaryOpId::Rsqrt);
    patterns.add<UnaryPattern<math::ErfOp>>(&getContext(), UnaryOpId::Erf);
    patterns.add<UnaryPattern<math::FloorOp>>(&getContext(), UnaryOpId::Floor);
    patterns.add<UnaryPattern<math::CeilOp>>(&getContext(), UnaryOpId::Ceil);
    patterns.add<UnaryPattern<tt::PreciseSqrtOp>>(&getContext(),
                                                  UnaryOpId::PreciseSqrt);
    patterns.add<ExternElementwisePattern>(&getContext());
    patterns.add<TMEMLoadPattern, TMEMStorePattern, TMEMCopyPattern,
                 TCGen5MMAPattern>(&getContext(), &scratch);
    patterns.add<TCGen5CommitPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      llvm::errs() << "FpSanitizer error: Failed to apply patterns\n";
      signalPassFailure();
    }

    getOperation()->walk([&](tt::ExternElementwiseOp op) {
      if (!externInvolvesFloatLike(op))
        return WalkResult::advance();

      hasUnsupportedOperations = true;
      llvm::errs()
          << "FpSanitizer error: Unsupported extern_elementwise: symbol="
          << op.getSymbol() << ", pure=" << op.getPure()
          << ", num_operands=" << op.getNumOperands() << ", result_ty=";
      op.getType().print(llvm::errs());
      llvm::errs() << ", operand_tys=(";
      llvm::interleaveComma(op.getOperandTypes(), llvm::errs(),
                            [&](Type ty) { ty.print(llvm::errs()); });
      llvm::errs() << ")\n";
      return WalkResult::interrupt();
    });
    if (hasUnsupportedOperations)
      signalPassFailure();

    // TODO: Remove unused tmem usages. This requires unwiring them from the
    // warp specialize partitions.
  }
};

} // namespace

} // namespace instrument
} // namespace triton
} // namespace mlir
