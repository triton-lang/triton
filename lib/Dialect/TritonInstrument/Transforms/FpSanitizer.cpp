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
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <numeric>
#include <optional>
#include <vector>

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

constexpr StringRef kFpsanScratchAttr = "tritoninstrument.fpsan_scratch";
constexpr bool kUseLoopDotEmulation = true;

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

Type getTypeWithElement(Type ty, Type elemTy) {
  if (auto ranked = dyn_cast<RankedTensorType>(ty))
    return RankedTensorType::get(ranked.getShape(), elemTy,
                                 ranked.getEncoding());
  if (auto vec = dyn_cast<VectorType>(ty))
    return VectorType::get(vec.getShape(), elemTy, vec.getScalableDims());
  if (auto shaped = dyn_cast<ShapedType>(ty))
    return shaped.clone(elemTy);
  return elemTy;
}

SmallVector<unsigned> insertOrderAtEnd(ArrayRef<unsigned> order,
                                       unsigned axis) {
  SmallVector<unsigned> newOrder;
  newOrder.reserve(order.size() + 1);
  for (unsigned dim : order)
    newOrder.push_back(dim < axis ? dim : dim + 1);
  newOrder.push_back(axis);
  return newOrder;
}

ttg::CGAEncodingAttr expandCGAEncoding(ttg::CGAEncodingAttr enc,
                                       unsigned axis) {
  auto perCGA = llvm::to_vector(enc.getCTAsPerCGA());
  auto splitNum = llvm::to_vector(enc.getCTASplitNum());
  auto order = llvm::to_vector(enc.getCTAOrder());
  perCGA.insert(perCGA.begin() + axis, 1);
  splitNum.insert(splitNum.begin() + axis, 1);
  auto newOrder = insertOrderAtEnd(order, axis);
  return ttg::CGAEncodingAttr::fromSplitParams(enc.getContext(), perCGA,
                                               splitNum, newOrder);
}

ttg::BlockedEncodingAttr expandBlockedEncoding(ttg::BlockedEncodingAttr enc,
                                               ArrayRef<int64_t> shape,
                                               unsigned axis) {
  auto sizePerThread = llvm::to_vector(enc.getSizePerThread());
  auto threadsPerWarp = llvm::to_vector(enc.getThreadsPerWarp());
  auto warpsPerCTA = llvm::to_vector(enc.getWarpsPerCTA());
  sizePerThread.insert(sizePerThread.begin() + axis, 1);
  threadsPerWarp.insert(threadsPerWarp.begin() + axis, 1);
  warpsPerCTA.insert(warpsPerCTA.begin() + axis, 1);
  auto order = insertOrderAtEnd(enc.getOrder(), axis);
  auto cga = expandCGAEncoding(enc.getCGALayout(), axis);
  return ttg::BlockedEncodingAttr::get(enc.getContext(), sizePerThread,
                                       threadsPerWarp, warpsPerCTA, order, cga);
}

ttg::BlockedEncodingAttr
inferCoalescedBlockedEncoding(Operation *op, RankedTensorType tensorType,
                              triton::ModuleAxisInfoAnalysis &axisInfo) {
  auto fallback = cast<ttg::BlockedEncodingAttr>(tensorType.getEncoding());
  int numWarps = ttg::lookupNumWarps(op);
  int threadsPerWarp =
      ttg::TritonGPUDialect::getThreadsPerWarp(op->getParentOfType<ModuleOp>());
  int numCTAs = ttg::lookupNumCTAs(op);
  auto cgaLayout = ttg::CGAEncodingAttr::get1CTALayout(op->getContext(),
                                                       tensorType.getRank());
  auto shapePerCTA =
      ttg::getShapePerCTA(cgaLayout.getCTASplitNum(), tensorType.getShape());
  auto ptr = getMemAccessPtr(op);
  if (!ptr || !axisInfo.getAxisInfo(ptr))
    return fallback;
  if (ptr.getDefiningOp()) {
    for (Operation *use : mlir::getSlice(op)) {
      Value val = getMemAccessPtr(use);
      if (!val)
        continue;
      if (!axisInfo.getAxisInfo(val))
        return fallback;
    }
  }
  return ttg::buildCoalescedEncoding(op->getContext(), axisInfo, op, numWarps,
                                     threadsPerWarp, cgaLayout, shapePerCTA);
}

Value loadScratchMemoryOptimized(PatternRewriter &rewriter, Location loc,
                                 Value alloc, RankedTensorType tensorType,
                                 triton::ModuleAxisInfoAnalysis *axisInfo) {
  Value scratchVal = createLoadScratchMemory(rewriter, loc, alloc, tensorType);
  if (auto loadOp = scratchVal.getDefiningOp<tt::LoadOp>()) {
    loadOp->setAttr(kFpsanScratchAttr, UnitAttr::get(rewriter.getContext()));
  }
  if (!scratchVal || !axisInfo)
    return scratchVal;
  if (!isa<ttg::BlockedEncodingAttr>(tensorType.getEncoding()))
    return scratchVal;
  auto loadOp = scratchVal.getDefiningOp<tt::LoadOp>();
  if (!loadOp)
    return scratchVal;
  auto optLayout = inferCoalescedBlockedEncoding(loadOp, tensorType, *axisInfo);
  if (optLayout == tensorType.getEncoding())
    return scratchVal;
  rewriter.eraseOp(loadOp);
  auto optType = RankedTensorType::get(tensorType.getShape(),
                                       tensorType.getElementType(), optLayout);
  scratchVal = createLoadScratchMemory(rewriter, loc, alloc, optType);
  if (auto newLoad = scratchVal.getDefiningOp<tt::LoadOp>()) {
    newLoad->setAttr(kFpsanScratchAttr, UnitAttr::get(rewriter.getContext()));
  }
  return scratchVal;
}

Operation *storeScratchMemoryOptimized(
    PatternRewriter &rewriter, Location loc, Value alloc, Value tensor,
    RankedTensorType tensorType, triton::ModuleAxisInfoAnalysis *axisInfo) {
  Operation *storeOp =
      createStoreScratchMemory(rewriter, loc, alloc, tensor, tensorType);
  if (storeOp) {
    storeOp->setAttr(kFpsanScratchAttr, UnitAttr::get(rewriter.getContext()));
  }
  if (!storeOp || !axisInfo)
    return storeOp;
  if (!isa<ttg::BlockedEncodingAttr>(tensorType.getEncoding()))
    return storeOp;
  auto optLayout =
      inferCoalescedBlockedEncoding(storeOp, tensorType, *axisInfo);
  if (optLayout == tensorType.getEncoding())
    return storeOp;
  rewriter.eraseOp(storeOp);
  auto optType = RankedTensorType::get(tensorType.getShape(),
                                       tensorType.getElementType(), optLayout);
  Value src = tensor;
  if (src.getType() != optType) {
    src = ttg::ConvertLayoutOp::create(rewriter, loc, optType, src).getResult();
  }
  storeOp = createStoreScratchMemory(rewriter, loc, alloc, src, optType);
  if (storeOp) {
    storeOp->setAttr(kFpsanScratchAttr, UnitAttr::get(rewriter.getContext()));
  }
  return storeOp;
}

Value castIntValueToType(PatternRewriter &rewriter, Location loc, Value v,
                         Type targetTy) {
  if (v.getType() == targetTy)
    return v;
  unsigned srcWidth = getIntBitwidth(v.getType());
  unsigned dstWidth = getIntBitwidth(targetTy);
  if (dstWidth > srcWidth)
    return arith::ExtUIOp::create(rewriter, loc, targetTy, v);
  return arith::TruncIOp::create(rewriter, loc, targetTy, v);
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

Value reduceAdd(PatternRewriter &rewriter, Location loc, Value tensor,
                int axis) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  auto reduceOp =
      tt::ReduceOp::create(rewriter, loc, std::vector<Value>{tensor}, axis);
  auto &block = reduceOp.getRegion().emplaceBlock();
  block.addArguments({tensorTy.getElementType(), tensorTy.getElementType()},
                     {loc, loc});
  rewriter.setInsertionPointToStart(&block);
  Value sum = arith::AddIOp::create(rewriter, loc, block.getArgument(0),
                                    block.getArgument(1));
  tt::ReduceReturnOp::create(rewriter, loc, sum);
  return reduceOp->getResult(0);
}

Value emulateDotLoop(PatternRewriter &rewriter, Location loc, Value aI,
                     Value bI, Value accInitI, int64_t m, int64_t k, int64_t n,
                     ttg::BlockedEncodingAttr accLayout) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto accTy = cast<RankedTensorType>(accInitI.getType());
  auto accElemTy = accTy.getElementType();
  auto idxTy = rewriter.getI32Type();
  auto aIdxTy = RankedTensorType::get({m, 1}, idxTy, accLayout);
  auto bIdxTy = RankedTensorType::get({1, n}, idxTy, accLayout);
  auto aSliceTy = RankedTensorType::get({m, 1}, accElemTy, accLayout);
  auto bSliceTy = RankedTensorType::get({1, n}, accElemTy, accLayout);
  auto fullTy = RankedTensorType::get({m, n}, accElemTy, accLayout);

  Value zero =
      arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
  Value upper =
      arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(k));
  Value step =
      arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));
  auto forOp = scf::ForOp::create(rewriter, loc, zero, upper, step, accInitI);
  rewriter.setInsertionPointToStart(forOp.getBody());
  Value iv = forOp.getInductionVar();
  Value kI32 =
      arith::IndexCastOp::create(rewriter, loc, rewriter.getI32Type(), iv);

  Value aIdx = tt::SplatOp::create(rewriter, loc, aIdxTy, kI32);
  Value bIdx = tt::SplatOp::create(rewriter, loc, bIdxTy, kI32);
  Value aSlice = tt::GatherOp::create(rewriter, loc, aSliceTy, aI, aIdx,
                                      rewriter.getI32IntegerAttr(1), UnitAttr())
                     .getResult();
  Value bSlice = tt::GatherOp::create(rewriter, loc, bSliceTy, bI, bIdx,
                                      rewriter.getI32IntegerAttr(0), UnitAttr())
                     .getResult();
  Value aFull = tt::BroadcastOp::create(rewriter, loc, fullTy, aSlice);
  Value bFull = tt::BroadcastOp::create(rewriter, loc, fullTy, bSlice);
  Value mul = arith::MulIOp::create(rewriter, loc, aFull, bFull);
  Value acc = forOp.getRegionIterArgs()[0];
  Value next = arith::AddIOp::create(rewriter, loc, acc, mul);
  scf::YieldOp::create(rewriter, loc, next);

  rewriter.setInsertionPointAfter(forOp);
  return forOp.getResult(0);
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
  getDefaultBlockedEncoding(PatternRewriter &rewriter,
                            ArrayRef<int64_t> shape) {
    int numWarps =
        ttg::lookupNumWarps(rewriter.getInsertionBlock()->getParent());
    int threadsPerWarp = ttg::lookupThreadsPerWarp(rewriter);
    int numCTAs =
        ttg::lookupNumCTAs(rewriter.getInsertionBlock()->getParentOp());
    return ttg::getDefaultBlockedEncoding(rewriter.getContext(), shape,
                                          numWarps, threadsPerWarp, numCTAs);
  }

  static ttg::BlockedEncodingAttr
  getOptimizedBlockedEncoding(PatternRewriter &rewriter,
                              ArrayRef<int64_t> shape, Type elemType) {
    auto base = getDefaultBlockedEncoding(rewriter, shape);
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
      auto layout = getOptimizedBlockedEncoding(rewriter, memTy.getShape(),
                                                memTy.getElementType());
      auto tensorTy = RankedTensorType::get(memTy.getShape(),
                                            memTy.getElementType(), layout);

      int64_t elSize = memTy.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t alignment = std::max<int64_t>(elSize, 16);
      int64_t sizeInBytes = getNumElements(memTy.getShape()) * elSize;
      auto ptrTy = triton::getPointerType(memTy.getElementType());
      auto allocOp = ttg::GlobalScratchAllocOp::create(rewriter, loc, ptrTy,
                                                       sizeInBytes, alignment);
      allocOp->setDiscardableAttr("tt.divisibility",
                                  rewriter.getI64IntegerAttr(alignment));
      Value ptr = allocOp.getResult();

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
          if (!storeScratchMemoryOptimized(rewriter, loc, ptr, init, tensorTy,
                                           nullptr))
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
      auto layout = getOptimizedBlockedEncoding(rewriter, memTy.getShape(),
                                                memTy.getElementType());
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
      auto layout = getOptimizedBlockedEncoding(rewriter, memTy.getShape(),
                                                memTy.getElementType());
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

      auto layout = getOptimizedBlockedEncoding(rewriter, memTy.getShape(),
                                                memTy.getElementType());
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
    auto accElem = cast<IntegerType>(getElementType(cI.getType()));
    aI = castIntValueToType(rewriter, loc, aI,
                            getTypeWithElement(aI.getType(), accElem));
    bI = castIntValueToType(rewriter, loc, bI,
                            getTypeWithElement(bI.getType(), accElem));
    auto dotI =
        tt::DotOp::create(rewriter, loc, aI, bI, cI, op.getInputPrecision(),
                          op.getMaxNumImpreciseAcc());
    auto resF = bitcastToFloat(rewriter, loc, dotI, op.getType());
    rewriter.replaceOp(op, resF);
    return success();
  }
};

struct TMEMLoadPattern : public OpRewritePattern<ttng::TMEMLoadOp> {
  TMEMLoadPattern(MLIRContext *ctx, TmemScratchManager *scratch,
                  triton::ModuleAxisInfoAnalysis *axisInfo)
      : OpRewritePattern(ctx), scratch(scratch), axisInfo(axisInfo) {}

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
      Value scratchVal = loadScratchMemoryOptimized(rewriter, loc, info->ptr,
                                                    info->tensorType, axisInfo);
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
  triton::ModuleAxisInfoAnalysis *axisInfo;
};

struct TMEMStorePattern : public OpRewritePattern<ttng::TMEMStoreOp> {
  TMEMStorePattern(MLIRContext *ctx, TmemScratchManager *scratch,
                   triton::ModuleAxisInfoAnalysis *axisInfo)
      : OpRewritePattern(ctx), scratch(scratch), axisInfo(axisInfo) {}

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
      if (!storeScratchMemoryOptimized(rewriter, loc, info->ptr, src,
                                       info->tensorType, axisInfo))
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
  triton::ModuleAxisInfoAnalysis *axisInfo;
};

struct TMEMCopyPattern : public OpRewritePattern<ttng::TMEMCopyOp> {
  TMEMCopyPattern(MLIRContext *ctx, TmemScratchManager *scratch,
                  triton::ModuleAxisInfoAnalysis *axisInfo)
      : OpRewritePattern(ctx), scratch(scratch), axisInfo(axisInfo) {}

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
    if (!storeScratchMemoryOptimized(rewriter, loc, info->ptr, srcReg,
                                     info->tensorType, axisInfo))
      return failure();

    if (Value barrier = op.getBarrier()) {
      ttng::ArriveBarrierOp::create(rewriter, loc, barrier, 1, Value());
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  TmemScratchManager *scratch;
  triton::ModuleAxisInfoAnalysis *axisInfo;
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
  TCGen5MMAPattern(MLIRContext *ctx, TmemScratchManager *scratch,
                   triton::ModuleAxisInfoAnalysis *axisInfo)
      : OpRewritePattern(ctx), scratch(scratch), axisInfo(axisInfo) {}

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

    Value accReg = loadScratchMemoryOptimized(rewriter, loc, dInfo->ptr,
                                              dInfo->tensorType, nullptr);
    if (!accReg)
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

    auto accLayout =
        dyn_cast<ttg::BlockedEncodingAttr>(dInfo->tensorType.getEncoding());
    if (!accLayout)
      return failure();

    std::optional<ScratchInfo> aInfo;
    if (aIsTmem) {
      aInfo = scratch->getOrCreate(op.getA(), rewriter, scope);
      if (!aInfo)
        return failure();
    }
    Value aReg;
    auto aLayout =
        aIsTmem
            ? cast<ttg::BlockedEncodingAttr>(aInfo->tensorType.getEncoding())
            : TmemScratchManager::getOptimizedBlockedEncoding(
                  rewriter, aMemTy.getShape(), aMemTy.getElementType());
    auto aRegTy = RankedTensorType::get(aMemTy.getShape(),
                                        aMemTy.getElementType(), aLayout);
    if (aIsTmem) {
      aReg = loadScratchMemoryOptimized(rewriter, loc, aInfo->ptr,
                                        aInfo->tensorType, nullptr);
      if (!aReg)
        return failure();
      if (aReg.getType() != aRegTy) {
        aReg = ttg::ConvertLayoutOp::create(rewriter, loc, aRegTy, aReg)
                   .getResult();
      }
    } else {
      aReg = ttg::LocalLoadOp::create(rewriter, loc, aRegTy, op.getA(), Value())
                 .getResult();
    }

    std::optional<ScratchInfo> bInfo;
    if (bIsTmem) {
      bInfo = scratch->getOrCreate(op.getB(), rewriter, scope);
      if (!bInfo)
        return failure();
    }
    Value bReg;
    auto bLayout =
        bIsTmem
            ? cast<ttg::BlockedEncodingAttr>(bInfo->tensorType.getEncoding())
            : TmemScratchManager::getOptimizedBlockedEncoding(
                  rewriter, bMemTy.getShape(), bMemTy.getElementType());
    auto bRegTy = RankedTensorType::get(bMemTy.getShape(),
                                        bMemTy.getElementType(), bLayout);
    if (bIsTmem) {
      bReg = loadScratchMemoryOptimized(rewriter, loc, bInfo->ptr,
                                        bInfo->tensorType, nullptr);
      if (!bReg)
        return failure();
      if (bReg.getType() != bRegTy) {
        bReg = ttg::ConvertLayoutOp::create(rewriter, loc, bRegTy, bReg)
                   .getResult();
      }
    } else {
      bReg = ttg::LocalLoadOp::create(rewriter, loc, bRegTy, op.getB(), Value())
                 .getResult();
    }

    Value accI = bitcastToInt(rewriter, loc, accReg);
    auto accElem = cast<IntegerType>(getElementType(accI.getType()));
    Value sum;
    if (kUseLoopDotEmulation) {
      Value aI = bitcastToInt(rewriter, loc, aReg);
      Value bI = bitcastToInt(rewriter, loc, bReg);
      aI = castIntValueToType(rewriter, loc, aI,
                              getTypeWithElement(aI.getType(), accElem));
      bI = castIntValueToType(rewriter, loc, bI,
                              getTypeWithElement(bI.getType(), accElem));
      Value zero = getIntConstantLike(rewriter, loc, accI, 0);
      sum = emulateDotLoop(rewriter, loc, aI, bI, zero, m, k, n, accLayout);
    } else {
      auto fullShape = SmallVector<int64_t>{m, k, n};
      auto fullEnc = expandBlockedEncoding(accLayout, fullShape, /*axis=*/1);
      auto *ctx = rewriter.getContext();
      auto aSliceEnc = ttg::SliceEncodingAttr::get(ctx, /*dim=*/2, fullEnc);
      auto bSliceEnc = ttg::SliceEncodingAttr::get(ctx, /*dim=*/0, fullEnc);

      auto aSliceTy = RankedTensorType::get(aMemTy.getShape(),
                                            aMemTy.getElementType(), aSliceEnc);
      Value aSlice = aReg;
      if (aSlice.getType() != aSliceTy) {
        aSlice = ttg::ConvertLayoutOp::create(rewriter, loc, aSliceTy, aSlice)
                     .getResult();
      }
      Value aI = bitcastToInt(rewriter, loc, aSlice);

      auto bSliceTy = RankedTensorType::get(bMemTy.getShape(),
                                            bMemTy.getElementType(), bSliceEnc);
      Value bSlice = bReg;
      if (bSlice.getType() != bSliceTy) {
        bSlice = ttg::ConvertLayoutOp::create(rewriter, loc, bSliceTy, bSlice)
                     .getResult();
      }
      Value bI = bitcastToInt(rewriter, loc, bSlice);

      aI = castIntValueToType(rewriter, loc, aI,
                              getTypeWithElement(aI.getType(), accElem));
      bI = castIntValueToType(rewriter, loc, bI,
                              getTypeWithElement(bI.getType(), accElem));

      auto elemTy = accElem;
      auto aExpTy = RankedTensorType::get({m, k, 1}, elemTy, fullEnc);
      auto bExpTy = RankedTensorType::get({1, k, n}, elemTy, fullEnc);
      auto fullTy = RankedTensorType::get({m, k, n}, elemTy, fullEnc);

      Value aExp =
          tt::ExpandDimsOp::create(rewriter, loc, aExpTy, aI, 2).getResult();
      Value bExp =
          tt::ExpandDimsOp::create(rewriter, loc, bExpTy, bI, 0).getResult();
      Value aFull =
          tt::BroadcastOp::create(rewriter, loc, fullTy, aExp).getResult();
      Value bFull =
          tt::BroadcastOp::create(rewriter, loc, fullTy, bExp).getResult();
      Value mul = arith::MulIOp::create(rewriter, loc, aFull, bFull);
      sum = reduceAdd(rewriter, loc, mul, /*axis=*/1);
      auto sumTy = RankedTensorType::get({m, n}, elemTy, accLayout);
      if (sum.getType() != sumTy) {
        sum =
            ttg::ConvertLayoutOp::create(rewriter, loc, sumTy, sum).getResult();
      }
    }
    Value useDInt =
        arith::ExtUIOp::create(rewriter, loc, accElem, op.getUseD());
    Value useDMask =
        tt::SplatOp::create(rewriter, loc, accI.getType(), useDInt);
    Value accInitI = arith::MulIOp::create(rewriter, loc, accI, useDMask);
    Value outI = arith::AddIOp::create(rewriter, loc, sum, accInitI);

    Value predInt =
        arith::ExtUIOp::create(rewriter, loc, accElem, op.getPred());
    Value predMask =
        tt::SplatOp::create(rewriter, loc, accI.getType(), predInt);
    Value oneI = getIntConstantLike(rewriter, loc, accI, 1);
    Value predInv = arith::SubIOp::create(rewriter, loc, oneI, predMask);
    Value outMasked = arith::MulIOp::create(rewriter, loc, outI, predMask);
    Value accMasked = arith::MulIOp::create(rewriter, loc, accI, predInv);
    Value outSelI = arith::AddIOp::create(rewriter, loc, outMasked, accMasked);
    Value out = bitcastToFloat(rewriter, loc, outSelI, accReg.getType());
    if (out.getType() != dInfo->tensorType) {
      out = ttg::ConvertLayoutOp::create(rewriter, loc, dInfo->tensorType, out)
                .getResult();
    }
    if (!storeScratchMemoryOptimized(rewriter, loc, dInfo->ptr, out,
                                     dInfo->tensorType, nullptr))
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
  triton::ModuleAxisInfoAnalysis *axisInfo;
};

struct TCGen5MMAScaledPattern
    : public OpRewritePattern<ttng::TCGen5MMAScaledOp> {
  TCGen5MMAScaledPattern(MLIRContext *ctx, TmemScratchManager *scratch,
                         triton::ModuleAxisInfoAnalysis *axisInfo)
      : OpRewritePattern(ctx), scratch(scratch), axisInfo(axisInfo) {}

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

    bool aIsTmem = isa<ttng::TensorMemorySpaceAttr>(aMemTy.getMemorySpace());
    bool bIsTmem = isa<ttng::TensorMemorySpaceAttr>(bMemTy.getMemorySpace());

    if ((aIsTmem && aMemTy.getRank() != 2) ||
        (bIsTmem && bMemTy.getRank() != 2) || dMemTy.getRank() != 2)
      return failure();

    Value accReg = loadScratchMemoryOptimized(rewriter, loc, dInfo->ptr,
                                              dInfo->tensorType, nullptr);
    if (!accReg)
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

    auto accLayout =
        dyn_cast<ttg::BlockedEncodingAttr>(dInfo->tensorType.getEncoding());
    if (!accLayout)
      return failure();

    std::optional<ScratchInfo> aInfo;
    if (aIsTmem) {
      aInfo = scratch->getOrCreate(op.getA(), rewriter, scope);
      if (!aInfo)
        return failure();
    }
    Value aReg;
    auto aLayout =
        aIsTmem
            ? cast<ttg::BlockedEncodingAttr>(aInfo->tensorType.getEncoding())
            : TmemScratchManager::getOptimizedBlockedEncoding(
                  rewriter, aMemTy.getShape(), aMemTy.getElementType());
    auto aRegTy = RankedTensorType::get(aMemTy.getShape(),
                                        aMemTy.getElementType(), aLayout);
    if (aIsTmem) {
      aReg = loadScratchMemoryOptimized(rewriter, loc, aInfo->ptr,
                                        aInfo->tensorType, nullptr);
      if (!aReg)
        return failure();
      if (aReg.getType() != aRegTy) {
        aReg = ttg::ConvertLayoutOp::create(rewriter, loc, aRegTy, aReg)
                   .getResult();
      }
    } else {
      aReg = ttg::LocalLoadOp::create(rewriter, loc, aRegTy, op.getA(), Value())
                 .getResult();
    }

    std::optional<ScratchInfo> bInfo;
    if (bIsTmem) {
      bInfo = scratch->getOrCreate(op.getB(), rewriter, scope);
      if (!bInfo)
        return failure();
    }
    Value bReg;
    auto bLayout =
        bIsTmem
            ? cast<ttg::BlockedEncodingAttr>(bInfo->tensorType.getEncoding())
            : TmemScratchManager::getOptimizedBlockedEncoding(
                  rewriter, bMemTy.getShape(), bMemTy.getElementType());
    auto bRegTy = RankedTensorType::get(bMemTy.getShape(),
                                        bMemTy.getElementType(), bLayout);
    if (bIsTmem) {
      bReg = loadScratchMemoryOptimized(rewriter, loc, bInfo->ptr,
                                        bInfo->tensorType, nullptr);
      if (!bReg)
        return failure();
      if (bReg.getType() != bRegTy) {
        bReg = ttg::ConvertLayoutOp::create(rewriter, loc, bRegTy, bReg)
                   .getResult();
      }
    } else {
      bReg = ttg::LocalLoadOp::create(rewriter, loc, bRegTy, op.getB(), Value())
                 .getResult();
    }

    Value accI = bitcastToInt(rewriter, loc, accReg);
    auto accElem = cast<IntegerType>(getElementType(accI.getType()));
    Value sum;
    if (kUseLoopDotEmulation) {
      Value aI = bitcastToInt(rewriter, loc, aReg);
      Value bI = bitcastToInt(rewriter, loc, bReg);
      aI = castIntValueToType(rewriter, loc, aI,
                              getTypeWithElement(aI.getType(), accElem));
      bI = castIntValueToType(rewriter, loc, bI,
                              getTypeWithElement(bI.getType(), accElem));
      Value zero = getIntConstantLike(rewriter, loc, accI, 0);
      sum = emulateDotLoop(rewriter, loc, aI, bI, zero, m, k, n, accLayout);
    } else {
      auto fullShape = SmallVector<int64_t>{m, k, n};
      auto fullEnc = expandBlockedEncoding(accLayout, fullShape, /*axis=*/1);
      auto *ctx = rewriter.getContext();
      auto aSliceEnc = ttg::SliceEncodingAttr::get(ctx, /*dim=*/2, fullEnc);
      auto bSliceEnc = ttg::SliceEncodingAttr::get(ctx, /*dim=*/0, fullEnc);

      auto aSliceTy = RankedTensorType::get(aMemTy.getShape(),
                                            aMemTy.getElementType(), aSliceEnc);
      Value aSlice = aReg;
      if (aSlice.getType() != aSliceTy) {
        aSlice = ttg::ConvertLayoutOp::create(rewriter, loc, aSliceTy, aSlice)
                     .getResult();
      }
      Value aI = bitcastToInt(rewriter, loc, aSlice);

      auto bSliceTy = RankedTensorType::get(bMemTy.getShape(),
                                            bMemTy.getElementType(), bSliceEnc);
      Value bSlice = bReg;
      if (bSlice.getType() != bSliceTy) {
        bSlice = ttg::ConvertLayoutOp::create(rewriter, loc, bSliceTy, bSlice)
                     .getResult();
      }
      Value bI = bitcastToInt(rewriter, loc, bSlice);

      aI = castIntValueToType(rewriter, loc, aI,
                              getTypeWithElement(aI.getType(), accElem));
      bI = castIntValueToType(rewriter, loc, bI,
                              getTypeWithElement(bI.getType(), accElem));

      auto elemTy = accElem;
      auto aExpTy = RankedTensorType::get({m, k, 1}, elemTy, fullEnc);
      auto bExpTy = RankedTensorType::get({1, k, n}, elemTy, fullEnc);
      auto fullTy = RankedTensorType::get({m, k, n}, elemTy, fullEnc);

      Value aExp =
          tt::ExpandDimsOp::create(rewriter, loc, aExpTy, aI, 2).getResult();
      Value bExp =
          tt::ExpandDimsOp::create(rewriter, loc, bExpTy, bI, 0).getResult();
      Value aFull =
          tt::BroadcastOp::create(rewriter, loc, fullTy, aExp).getResult();
      Value bFull =
          tt::BroadcastOp::create(rewriter, loc, fullTy, bExp).getResult();
      Value mul = arith::MulIOp::create(rewriter, loc, aFull, bFull);
      sum = reduceAdd(rewriter, loc, mul, /*axis=*/1);
      auto sumTy = RankedTensorType::get({m, n}, elemTy, accLayout);
      if (sum.getType() != sumTy) {
        sum =
            ttg::ConvertLayoutOp::create(rewriter, loc, sumTy, sum).getResult();
      }
    }
    Value useDInt =
        arith::ExtUIOp::create(rewriter, loc, accElem, op.getUseD());
    Value useDMask =
        tt::SplatOp::create(rewriter, loc, accI.getType(), useDInt);
    Value accInitI = arith::MulIOp::create(rewriter, loc, accI, useDMask);
    Value outI = arith::AddIOp::create(rewriter, loc, sum, accInitI);

    Value predInt =
        arith::ExtUIOp::create(rewriter, loc, accElem, op.getPred());
    Value predMask =
        tt::SplatOp::create(rewriter, loc, accI.getType(), predInt);
    Value oneI = getIntConstantLike(rewriter, loc, accI, 1);
    Value predInv = arith::SubIOp::create(rewriter, loc, oneI, predMask);
    Value outMasked = arith::MulIOp::create(rewriter, loc, outI, predMask);
    Value accMasked = arith::MulIOp::create(rewriter, loc, accI, predInv);
    Value outSelI = arith::AddIOp::create(rewriter, loc, outMasked, accMasked);
    Value out = bitcastToFloat(rewriter, loc, outSelI, accReg.getType());
    if (out.getType() != dInfo->tensorType) {
      out = ttg::ConvertLayoutOp::create(rewriter, loc, dInfo->tensorType, out)
                .getResult();
    }
    if (!storeScratchMemoryOptimized(rewriter, loc, dInfo->ptr, out,
                                     dInfo->tensorType, nullptr))
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
  triton::ModuleAxisInfoAnalysis *axisInfo;
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
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::gluon::GluonDialect>();
  }

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
                                                           &scratch, nullptr);
    patterns.add<TCGen5CommitPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }

    auto moduleOp = getOperation();
    triton::ModuleAxisInfoAnalysis axisInfo(moduleOp);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(moduleOp);
    SmallVector<std::pair<Operation *, Attribute>> layoutFixups;
    moduleOp.walk([&](Operation *op) {
      if (!op->hasAttr(kFpsanScratchAttr))
        return;
      Value ptr = getMemAccessPtr(op);
      if (!ptr)
        return;
      auto tensorType = dyn_cast<RankedTensorType>(ptr.getType());
      if (!tensorType || !isa<PointerType>(tensorType.getElementType()))
        return;
      int numWarps = ttg::lookupNumWarps(op);
      auto cgaLayout = ttg::getCGALayout(tensorType.getEncoding());
      auto shapePerCTA = ttg::getShapePerCTA(tensorType);
      auto layout =
          ttg::buildCoalescedEncoding(&getContext(), axisInfo, op, numWarps,
                                      threadsPerWarp, cgaLayout, shapePerCTA);
      if (layout == tensorType.getEncoding())
        return;
      layoutFixups.emplace_back(op, layout);
    });
    for (auto &entry : layoutFixups) {
      Operation *op = entry.first;
      if (!op || !op->getParentOp())
        continue;
      Operation *newOp = convertDistributedOpEncoding(entry.second, op);
      newOp->removeAttr(kFpsanScratchAttr);
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
