#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <optional>

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONINSTRUMENTFPSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

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
  if (auto ranked = dyn_cast<RankedTensorType>(ty))
    return RankedTensorType::get(ranked.getShape(), intElem,
                                 ranked.getEncoding());
  if (auto vec = dyn_cast<VectorType>(ty))
    return VectorType::get(vec.getShape(), intElem, vec.getScalableDims());
  if (auto shaped = dyn_cast<ShapedType>(ty))
    return shaped.clone(intElem);
  return intElem;
}

unsigned getIntBitwidth(Type ty) {
  auto elem = cast<IntegerType>(getElementType(ty));
  return elem.getWidth();
}

int64_t getNumElements(ArrayRef<int64_t> shape) {
  int64_t num = 1;
  for (int64_t dim : shape)
    num *= dim;
  return num;
}

Value bitcastToInt(PatternRewriter &rewriter, Location loc, Value v) {
  auto intTy = getIntTypeLike(v.getType());
  return tt::BitcastOp::create(rewriter, loc, intTy, v);
}

Value bitcastToFloat(PatternRewriter &rewriter, Location loc, Value v,
                     Type floatTy) {
  return tt::BitcastOp::create(rewriter, loc, floatTy, v);
}

Value getIntConstantLike(PatternRewriter &rewriter, Location loc, Value like,
                         int64_t value) {
  auto ty = like.getType();
  if (auto shaped = dyn_cast<ShapedType>(ty)) {
    auto elem = cast<IntegerType>(shaped.getElementType());
    auto attr =
        DenseElementsAttr::get(shaped, rewriter.getIntegerAttr(elem, value));
    return arith::ConstantOp::create(rewriter, loc, attr);
  }
  auto intTy = cast<IntegerType>(ty);
  return arith::ConstantOp::create(rewriter, loc,
                                   rewriter.getIntegerAttr(intTy, value));
}

Value fpsanPow2ModInv(PatternRewriter &rewriter, Location loc, Value input) {
  auto one = getIntConstantLike(rewriter, loc, input, 1);
  auto two = getIntConstantLike(rewriter, loc, input, 2);

  auto a = arith::OrIOp::create(rewriter, loc, input, one);
  Value x = a;

  unsigned bitwidth = getIntBitwidth(input.getType());
  unsigned iters = 0;
  unsigned bits = 1;
  while (bits < bitwidth) {
    bits *= 2;
    iters += 1;
  }

  for (unsigned i = 0; i < iters; ++i) {
    auto ax = arith::MulIOp::create(rewriter, loc, a, x);
    auto twoMinusAx = arith::SubIOp::create(rewriter, loc, two, ax);
    x = arith::MulIOp::create(rewriter, loc, x, twoMinusAx);
  }
  return x;
}

Value fpsanFDiv(PatternRewriter &rewriter, Location loc, Value num, Value den) {
  auto numI = bitcastToInt(rewriter, loc, num);
  auto denI = bitcastToInt(rewriter, loc, den);
  auto inv = fpsanPow2ModInv(rewriter, loc, denI);
  auto resI = arith::MulIOp::create(rewriter, loc, numI, inv);
  return bitcastToFloat(rewriter, loc, resI, num.getType());
}

Value fpsanSRem(PatternRewriter &rewriter, Location loc, Value num, Value den) {
  auto numI = bitcastToInt(rewriter, loc, num);
  auto denI = bitcastToInt(rewriter, loc, den);
  auto one = getIntConstantLike(rewriter, loc, denI, 1);
  auto denSafe = arith::OrIOp::create(rewriter, loc, denI, one);
  auto resI = arith::RemSIOp::create(rewriter, loc, numI, denSafe);
  return bitcastToFloat(rewriter, loc, resI, num.getType());
}

struct ScratchInfo {
  Value ptr;
  RankedTensorType tensorType;
};

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
  return nullptr;
}

bool isValueDefinedInRegion(Value value, Region *region) {
  if (!region)
    return false;
  if (auto arg = dyn_cast<BlockArgument>(value))
    return arg.getOwner()->getParent() == region;
  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  return def->getParentRegion() == region;
}

class TmemScratchManager {
public:
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

  static ttg::BlockedEncodingAttr
  getDefaultScratchEncoding(PatternRewriter &rewriter,
                            ArrayRef<int64_t> shape) {
    int numWarps =
        ttg::lookupNumWarps(rewriter.getInsertionBlock()->getParent());
    int threadsPerWarp = ttg::lookupThreadsPerWarp(rewriter);
    int numCTAs =
        ttg::lookupNumCTAs(rewriter.getInsertionBlock()->getParentOp());
    return ttg::getDefaultBlockedEncoding(rewriter.getContext(), shape,
                                          numWarps, threadsPerWarp, numCTAs);
  }

  std::optional<ScratchInfo>
  getOrCreate(Value memdesc, PatternRewriter &rewriter, Region *scope) {
    if (!scope)
      return std::nullopt;

    if (auto arg = dyn_cast<BlockArgument>(memdesc)) {
      if (auto wsPartitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(
              arg.getOwner()->getParentOp())) {
        auto capture = wsPartitions.getParentOp()
                           .getExplicitCaptures()[arg.getArgNumber()];
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
      rewriter.setInsertionPointToStart(&scope->front());
      auto loc = alloc.getLoc();
      auto layout = getDefaultScratchEncoding(rewriter, memTy.getShape());
      auto tensorTy = RankedTensorType::get(memTy.getShape(),
                                            memTy.getElementType(), layout);

      int64_t elSize = memTy.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t sizeInBytes = getNumElements(memTy.getShape()) * elSize;
      auto ptrTy = triton::getPointerType(memTy.getElementType());
      Value ptr = ttg::GlobalScratchAllocOp::create(rewriter, loc, ptrTy,
                                                    sizeInBytes, elSize);

      if (Value src = alloc.getSrc()) {
        if (!isValueDefinedInRegion(src, scope))
          src = Value();
        if (src && src.getDefiningOp())
          rewriter.setInsertionPointAfter(src.getDefiningOp());
        Value init = src;
        if (init && init.getType() != tensorTy) {
          init = ttg::ConvertLayoutOp::create(rewriter, loc, tensorTy, init)
                     .getResult();
        }
        if (init) {
          if (!createStoreScratchMemory(rewriter, loc, ptr, init, tensorTy))
            return std::nullopt;
        }
      }

      ScratchInfo info{ptr, tensorTy};
      scratchMap[memdesc][scope] = info;
      return info;
    }

    if (auto subslice = memdesc.getDefiningOp<ttng::TMEMSubSliceOp>()) {
      auto baseInfo = getOrCreate(subslice.getSrc(), rewriter, scope);
      if (!baseInfo)
        return std::nullopt;

      auto baseTy = cast<ttg::MemDescType>(subslice.getSrc().getType());
      if (baseTy.getRank() != 2 || memTy.getRank() != 2)
        return std::nullopt;

      OpBuilder::InsertionGuard guard(rewriter);
      if (Operation *def = baseInfo->ptr.getDefiningOp();
          def && def->getParentRegion() == scope) {
        rewriter.setInsertionPointAfter(def);
      } else {
        rewriter.setInsertionPointToStart(&scope->front());
      }
      auto loc = subslice.getLoc();
      int64_t stride = baseTy.getShape().front();
      int64_t offset = subslice.getN();
      auto offsetVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(offset));
      auto strideVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(stride));
      auto offsetEls = arith::MulIOp::create(
          rewriter, loc, rewriter.getI32Type(), offsetVal, strideVal);
      auto ptr = tt::AddPtrOp::create(rewriter, loc, baseInfo->ptr.getType(),
                                      baseInfo->ptr, offsetEls);
      auto layout = getDefaultScratchEncoding(rewriter, memTy.getShape());
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
      if (Operation *def = baseInfo->ptr.getDefiningOp();
          def && def->getParentRegion() == scope) {
        rewriter.setInsertionPointAfter(def);
      } else {
        rewriter.setInsertionPointToStart(&scope->front());
      }
      auto loc = view.getLoc();
      Value idx = view.getIndex();
      if (!isValueDefinedInRegion(idx, scope)) {
        APInt value;
        if (!matchPattern(idx, m_ConstantInt(&value)))
          return std::nullopt;
        idx = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI32IntegerAttr(value.getSExtValue()));
      }
      idx = castToI32(rewriter, loc, idx);
      if (!idx)
        return std::nullopt;
      int64_t stride = getNumElements(baseTy.getShape().drop_front(1));
      auto strideVal = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(stride));
      auto offset = arith::MulIOp::create(rewriter, loc, rewriter.getI32Type(),
                                          idx, strideVal);
      auto ptr = tt::AddPtrOp::create(rewriter, loc, baseInfo->ptr.getType(),
                                      baseInfo->ptr, offset);
      auto layout = getDefaultScratchEncoding(rewriter, memTy.getShape());
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
      if (Operation *def = baseInfo->ptr.getDefiningOp();
          def && def->getParentRegion() == scope) {
        rewriter.setInsertionPointAfter(def);
      } else {
        rewriter.setInsertionPointToStart(&scope->front());
      }
      auto loc = view.getLoc();
      Value ptr = baseInfo->ptr;
      auto ptrTy = triton::getPointerType(memTy.getElementType());
      if (ptr.getType() != ptrTy) {
        ptr = tt::BitcastOp::create(rewriter, loc, ptrTy, ptr);
      }

      auto layout = getDefaultScratchEncoding(rewriter, memTy.getShape());
      auto tensorTy = RankedTensorType::get(memTy.getShape(),
                                            memTy.getElementType(), layout);

      ScratchInfo info{ptr, tensorTy};
      return info;
    }

    return std::nullopt;
  }

private:
  DenseMap<Value, DenseMap<Region *, ScratchInfo>> scratchMap;
};

Value createAsyncToken(PatternRewriter &rewriter, Location loc,
                       ValueRange deps) {
  return ttg::AsyncCommitGroupOp::create(rewriter, loc, deps).getResult();
}

Value createPointerTensorUnencoded2D(PatternRewriter &rewriter, Location loc,
                                     Value base, ArrayRef<int64_t> shape) {
  auto ptrTy = base.getType();
  auto ptrTensorTy = RankedTensorType::get(shape, ptrTy);
  Value ptrTensor = tt::SplatOp::create(rewriter, loc, ptrTensorTy, base);
  auto i32Ty = rewriter.getI32Type();
  auto offsetsTy = RankedTensorType::get(shape, i32Ty);

  auto dim0Ty = RankedTensorType::get({shape[0]}, i32Ty);
  auto range0 = tt::MakeRangeOp::create(rewriter, loc, dim0Ty, 0, shape[0]);
  auto stride0 = createConstIntTensor(rewriter, loc, 1, dim0Ty);
  auto off0 = arith::MulIOp::create(rewriter, loc, dim0Ty, range0, stride0);
  auto off0ExpTy = RankedTensorType::get({shape[0], 1}, i32Ty);
  auto off0Exp = tt::ExpandDimsOp::create(rewriter, loc, off0ExpTy, off0, 1);
  auto off0Full = tt::BroadcastOp::create(rewriter, loc, offsetsTy, off0Exp);
  ptrTensor =
      tt::AddPtrOp::create(rewriter, loc, ptrTensorTy, ptrTensor, off0Full);

  auto dim1Ty = RankedTensorType::get({shape[1]}, i32Ty);
  auto range1 = tt::MakeRangeOp::create(rewriter, loc, dim1Ty, 0, shape[1]);
  auto stride1 = createConstIntTensor(rewriter, loc, shape[0], dim1Ty);
  auto off1 = arith::MulIOp::create(rewriter, loc, dim1Ty, range1, stride1);
  auto off1ExpTy = RankedTensorType::get({1, shape[1]}, i32Ty);
  auto off1Exp = tt::ExpandDimsOp::create(rewriter, loc, off1ExpTy, off1, 0);
  auto off1Full = tt::BroadcastOp::create(rewriter, loc, offsetsTy, off1Exp);
  ptrTensor =
      tt::AddPtrOp::create(rewriter, loc, ptrTensorTy, ptrTensor, off1Full);

  return ptrTensor;
}

Value loadScratchUnencoded(PatternRewriter &rewriter, Location loc, Value base,
                           RankedTensorType resultTy) {
  auto shape = resultTy.getShape();
  auto ptrTensor = createPointerTensorUnencoded2D(rewriter, loc, base, shape);
  return tt::LoadOp::create(rewriter, loc, ptrTensor, CacheModifier::NONE,
                            EvictionPolicy::NORMAL, false);
}

Operation *storeScratchUnencoded(PatternRewriter &rewriter, Location loc,
                                 Value base, Value tensor,
                                 RankedTensorType tensorTy) {
  auto ptrTensor =
      createPointerTensorUnencoded2D(rewriter, loc, base, tensorTy.getShape());
  return tt::StoreOp::create(rewriter, loc, ptrTensor, tensor,
                             CacheModifier::NONE, EvictionPolicy::NORMAL);
}

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

struct DotPattern : public OpRewritePattern<tt::DotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(tt::DotOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    auto loc = op.getLoc();
    auto aI = bitcastToInt(rewriter, loc, op.getA());
    auto bI = bitcastToInt(rewriter, loc, op.getB());
    auto cI = bitcastToInt(rewriter, loc, op.getC());
    auto dotI =
        tt::DotOp::create(rewriter, loc, aI, bI, cI, op.getInputPrecision(),
                          op.getMaxNumImpreciseAcc());
    auto resF = bitcastToFloat(rewriter, loc, dotI, op.getType());
    rewriter.replaceOp(op, resF);
    return success();
  }
};

struct TMEMLoadPattern : public OpRewritePattern<ttng::TMEMLoadOp> {
  TMEMLoadPattern(MLIRContext *ctx, TmemScratchManager *scratch)
      : OpRewritePattern(ctx), scratch(scratch) {}

  LogicalResult matchAndRewrite(ttng::TMEMLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto scope = getScratchScopeRegion(op);
    auto info = scratch->getOrCreate(op.getSrc(), rewriter, scope);
    if (!info)
      return failure();

    auto loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    Value result;
    if (resultTy.getEncoding()) {
      Value scratchVal =
          createLoadScratchMemory(rewriter, loc, info->ptr, info->tensorType);
      if (!scratchVal)
        return failure();
      if (scratchVal.getType() != resultTy) {
        scratchVal =
            ttg::ConvertLayoutOp::create(rewriter, loc, resultTy, scratchVal)
                .getResult();
      }
      result = scratchVal;
    } else {
      if (resultTy.getRank() != 2)
        return failure();
      result = loadScratchUnencoded(rewriter, loc, info->ptr, resultTy);
    }

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
    if (srcTy.getEncoding()) {
      Value src = op.getSrc();
      if (src.getType() != info->tensorType) {
        src = ttg::ConvertLayoutOp::create(rewriter, loc, info->tensorType, src)
                  .getResult();
      }
      if (!createStoreScratchMemory(rewriter, loc, info->ptr, src,
                                    info->tensorType))
        return failure();
    } else {
      if (srcTy.getRank() != 2)
        return failure();
      storeScratchUnencoded(rewriter, loc, info->ptr, op.getSrc(), srcTy);
    }

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
    if (srcReg.getType() != info->tensorType) {
      srcReg =
          ttg::ConvertLayoutOp::create(rewriter, loc, info->tensorType, srcReg)
              .getResult();
    }
    if (!createStoreScratchMemory(rewriter, loc, info->ptr, srcReg,
                                  info->tensorType))
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
    auto *ctx = rewriter.getContext();

    bool aIsTmem = isa<ttng::TensorMemorySpaceAttr>(aMemTy.getMemorySpace());
    bool bIsTmem = isa<ttng::TensorMemorySpaceAttr>(bMemTy.getMemorySpace());

    if ((aIsTmem && aMemTy.getRank() != 2) ||
        (bIsTmem && bMemTy.getRank() != 2) || dMemTy.getRank() != 2)
      return failure();

    Value accReg =
        createLoadScratchMemory(rewriter, loc, dInfo->ptr, dInfo->tensorType);
    if (!accReg)
      return failure();

    auto accLayout = dInfo->tensorType.getEncoding();

    Value aReg;
    if (aIsTmem) {
      auto aInfo = scratch->getOrCreate(op.getA(), rewriter, scope);
      if (!aInfo)
        return failure();
      aReg =
          createLoadScratchMemory(rewriter, loc, aInfo->ptr, aInfo->tensorType);
      if (!aReg)
        return failure();
    } else {
      auto regTy = RankedTensorType::get(aMemTy.getShape(),
                                         aMemTy.getElementType(), accLayout);
      aReg = ttg::LocalLoadOp::create(rewriter, loc, regTy, op.getA(), Value())
                 .getResult();
    }

    Value bReg;
    if (bIsTmem) {
      auto bInfo = scratch->getOrCreate(op.getB(), rewriter, scope);
      if (!bInfo)
        return failure();
      bReg =
          createLoadScratchMemory(rewriter, loc, bInfo->ptr, bInfo->tensorType);
      if (!bReg)
        return failure();
    } else {
      auto regTy = RankedTensorType::get(bMemTy.getShape(),
                                         bMemTy.getElementType(), accLayout);
      bReg = ttg::LocalLoadOp::create(rewriter, loc, regTy, op.getB(), Value())
                 .getResult();
    }

    Value accI = bitcastToInt(rewriter, loc, accReg);
    Value zeroI = getIntConstantLike(rewriter, loc, accI, 0);
    Value zeroF = bitcastToFloat(rewriter, loc, zeroI, accReg.getType());
    Value accInit =
        arith::SelectOp::create(rewriter, loc, op.getUseD(), accReg, zeroF);
    auto aDotEnc = ttg::DotOperandEncodingAttr::get(ctx, 0, accLayout,
                                                    aMemTy.getElementType());
    auto bDotEnc = ttg::DotOperandEncodingAttr::get(ctx, 1, accLayout,
                                                    bMemTy.getElementType());
    auto aDotTy = RankedTensorType::get(aMemTy.getShape(),
                                        aMemTy.getElementType(), aDotEnc);
    auto bDotTy = RankedTensorType::get(bMemTy.getShape(),
                                        bMemTy.getElementType(), bDotEnc);
    Value aDot =
        ttg::ConvertLayoutOp::create(rewriter, loc, aDotTy, aReg).getResult();
    Value bDot =
        ttg::ConvertLayoutOp::create(rewriter, loc, bDotTy, bReg).getResult();

    Value aDotI = bitcastToInt(rewriter, loc, aDot);
    Value bDotI = bitcastToInt(rewriter, loc, bDot);
    Value accInitI = bitcastToInt(rewriter, loc, accInit);
    auto dotI = tt::DotOp::create(rewriter, loc, accInitI.getType(), aDotI,
                                  bDotI, accInitI, tt::InputPrecision::IEEE, 0);
    Value dotF = bitcastToFloat(rewriter, loc, dotI, accReg.getType());
    Value out =
        arith::SelectOp::create(rewriter, loc, op.getPred(), dotF, accReg);
    if (!createStoreScratchMemory(rewriter, loc, dInfo->ptr, out,
                                  dInfo->tensorType))
      return failure();

    if (!op.getBarriers().empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(op);
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

struct TCGen5MMAScaledPattern
    : public OpRewritePattern<ttng::TCGen5MMAScaledOp> {
  TCGen5MMAScaledPattern(MLIRContext *ctx, TmemScratchManager *scratch)
      : OpRewritePattern(ctx), scratch(scratch) {}

  LogicalResult matchAndRewrite(ttng::TCGen5MMAScaledOp op,
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
    auto *ctx = rewriter.getContext();

    bool aIsTmem = isa<ttng::TensorMemorySpaceAttr>(aMemTy.getMemorySpace());
    bool bIsTmem = isa<ttng::TensorMemorySpaceAttr>(bMemTy.getMemorySpace());

    if ((aIsTmem && aMemTy.getRank() != 2) ||
        (bIsTmem && bMemTy.getRank() != 2) || dMemTy.getRank() != 2)
      return failure();

    Value accReg =
        createLoadScratchMemory(rewriter, loc, dInfo->ptr, dInfo->tensorType);
    if (!accReg)
      return failure();

    auto accLayout = dInfo->tensorType.getEncoding();

    Value aReg;
    if (aIsTmem) {
      auto aInfo = scratch->getOrCreate(op.getA(), rewriter, scope);
      if (!aInfo)
        return failure();
      aReg =
          createLoadScratchMemory(rewriter, loc, aInfo->ptr, aInfo->tensorType);
      if (!aReg)
        return failure();
    } else {
      auto regTy = RankedTensorType::get(aMemTy.getShape(),
                                         aMemTy.getElementType(), accLayout);
      aReg = ttg::LocalLoadOp::create(rewriter, loc, regTy, op.getA(), Value())
                 .getResult();
    }

    Value bReg;
    if (bIsTmem) {
      auto bInfo = scratch->getOrCreate(op.getB(), rewriter, scope);
      if (!bInfo)
        return failure();
      bReg =
          createLoadScratchMemory(rewriter, loc, bInfo->ptr, bInfo->tensorType);
      if (!bReg)
        return failure();
    } else {
      auto regTy = RankedTensorType::get(bMemTy.getShape(),
                                         bMemTy.getElementType(), accLayout);
      bReg = ttg::LocalLoadOp::create(rewriter, loc, regTy, op.getB(), Value())
                 .getResult();
    }

    Value accI = bitcastToInt(rewriter, loc, accReg);
    Value zeroI = getIntConstantLike(rewriter, loc, accI, 0);
    Value zeroF = bitcastToFloat(rewriter, loc, zeroI, accReg.getType());
    Value accInit =
        arith::SelectOp::create(rewriter, loc, op.getUseD(), accReg, zeroF);
    auto aDotEnc = ttg::DotOperandEncodingAttr::get(ctx, 0, accLayout,
                                                    aMemTy.getElementType());
    auto bDotEnc = ttg::DotOperandEncodingAttr::get(ctx, 1, accLayout,
                                                    bMemTy.getElementType());
    auto aDotTy = RankedTensorType::get(aMemTy.getShape(),
                                        aMemTy.getElementType(), aDotEnc);
    auto bDotTy = RankedTensorType::get(bMemTy.getShape(),
                                        bMemTy.getElementType(), bDotEnc);
    Value aDot =
        ttg::ConvertLayoutOp::create(rewriter, loc, aDotTy, aReg).getResult();
    Value bDot =
        ttg::ConvertLayoutOp::create(rewriter, loc, bDotTy, bReg).getResult();

    Value aDotI = bitcastToInt(rewriter, loc, aDot);
    Value bDotI = bitcastToInt(rewriter, loc, bDot);
    Value accInitI = bitcastToInt(rewriter, loc, accInit);
    auto dotI = tt::DotOp::create(rewriter, loc, accInitI.getType(), aDotI,
                                  bDotI, accInitI, tt::InputPrecision::IEEE, 0);
    Value dotF = bitcastToFloat(rewriter, loc, dotI, accReg.getType());
    Value out =
        arith::SelectOp::create(rewriter, loc, op.getPred(), dotF, accReg);
    if (!createStoreScratchMemory(rewriter, loc, dInfo->ptr, out,
                                  dInfo->tensorType))
      return failure();

    if (!op.getBarriers().empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(op);
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

template <typename OpTy>
struct IdentityPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    rewriter.replaceOp(op, op.getOperand());
    return success();
  }
};

class FpSanitizerPass
    : public impl::TritonInstrumentFpSanitizerBase<FpSanitizerPass> {
public:
  void runOnOperation() override {
    TmemScratchManager scratch;
    RewritePatternSet patterns(&getContext());
    patterns.add<BinaryFloatToIntPattern<arith::AddFOp, arith::AddIOp>,
                 BinaryFloatToIntPattern<arith::SubFOp, arith::SubIOp>,
                 BinaryFloatToIntPattern<arith::MulFOp, arith::MulIOp>,
                 DivFOpPattern, PreciseDivFOpPattern, RemFOpPattern, FmaPattern,
                 DotPattern, IdentityPattern<math::ExpOp>,
                 IdentityPattern<math::LogOp>, IdentityPattern<math::Exp2Op>,
                 IdentityPattern<math::Log2Op>, IdentityPattern<math::CosOp>,
                 IdentityPattern<math::SinOp>, IdentityPattern<math::SqrtOp>,
                 IdentityPattern<math::RsqrtOp>, IdentityPattern<math::ErfOp>,
                 IdentityPattern<math::FloorOp>, IdentityPattern<math::CeilOp>,
                 IdentityPattern<tt::PreciseSqrtOp>>(&getContext());
    patterns.add<TMEMLoadPattern, TMEMStorePattern, TMEMCopyPattern,
                 TCGen5MMAPattern, TCGen5MMAScaledPattern>(&getContext(),
                                                           &scratch);
    patterns.add<TCGen5CommitPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }

    auto onlyUsedByWarpSpecialize = [](Value value) -> bool {
      for (OpOperand &use : value.getUses()) {
        if (isa<ttg::WarpSpecializeOp, ttg::MemDescIndexOp,
                ttng::TMEMSubSliceOp, ttg::MemDescReinterpretOp>(
                use.getOwner()))
          continue;
        return false;
      }
      return true;
    };

    SmallVector<Operation *> eraseOps;
    bool hasUnsupported = false;
    getOperation().walk([&](Operation *op) {
      if (isa<ttng::TMEMAllocOp, ttng::TMEMSubSliceOp>(op)) {
        if (op->use_empty()) {
          eraseOps.push_back(op);
        } else if (auto alloc = dyn_cast<ttng::TMEMAllocOp>(op)) {
          if (!onlyUsedByWarpSpecialize(alloc.getResult()))
            hasUnsupported = true;
        } else {
          hasUnsupported = true;
        }
        return;
      }
      if (auto view = dyn_cast<ttg::MemDescIndexOp>(op)) {
        auto memTy = dyn_cast<ttg::MemDescType>(view.getType());
        if (memTy && isa<ttng::TensorMemorySpaceAttr>(memTy.getMemorySpace()) &&
            view->use_empty()) {
          eraseOps.push_back(op);
        }
        return;
      }
      if (auto view = dyn_cast<ttg::MemDescReinterpretOp>(op)) {
        auto memTy = dyn_cast<ttg::MemDescType>(view.getType());
        if (memTy && isa<ttng::TensorMemorySpaceAttr>(memTy.getMemorySpace()) &&
            view->use_empty()) {
          eraseOps.push_back(op);
        }
        return;
      }
      if (isa<ttng::TMEMLoadOp, ttng::TMEMStoreOp, ttng::TMEMCopyOp,
              ttng::TCGen5MMAOp, ttng::TCGen5MMAScaledOp, ttng::TCGen5CommitOp>(
              op)) {
        hasUnsupported = true;
      }
    });

    for (Operation *op : eraseOps)
      op->erase();
    if (hasUnsupported)
      signalPassFailure();
  }
};

} // namespace

} // namespace instrument
} // namespace triton
} // namespace mlir
