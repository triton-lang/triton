#include "mlir/IR/Diagnostics.h"
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
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include <cassert>
#include <functional>

namespace mlir {
namespace triton {
namespace instrument {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONINSTRUMENTFPSANITIZER
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h.inc"

namespace {

Type getIntTypeLike(Type ty);
bool isFloatLike(Type ty) { return isa<FloatType>(getElementTypeOrSelf(ty)); }
bool isIntLike(Type ty) { return isa<IntegerType>(getElementTypeOrSelf(ty)); }

bool isNumericLike(Type ty) {
  Type elemTy = getElementTypeOrSelf(ty);
  return isa<FloatType>(elemTy) || isa<IntegerType>(elemTy);
}

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
constexpr int64_t kI8MmaM = 16;
constexpr int64_t kI8MmaN = 8;
constexpr int64_t kI8MmaK = 32;

bool supportsI8DotDecomposition(PatternRewriter &rewriter,
                                IntegerType accElem) {
  auto moduleOp =
      rewriter.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  if (getAMDArch(moduleOp))
    return false;
  return llvm::is_contained({16, 32, 64}, accElem.getWidth());
}

bool canUseI8MmaTile(int64_t m, int64_t n, int numWarps) {
  return m >= kI8MmaM && n >= kI8MmaN &&
         (m / kI8MmaM) * (n / kI8MmaN) >= numWarps;
}

std::pair<int64_t, int64_t> getMmaEmulationTileShape(PatternRewriter &rewriter,
                                                     int64_t m, int64_t n,
                                                     int64_t k,
                                                     IntegerType accElem) {
  std::pair<int64_t, int64_t> tile = {std::min<int64_t>(kTileM, m),
                                      std::min<int64_t>(kTileN, n)};
  int64_t numWarps =
      ttg::lookupNumWarps(rewriter.getInsertionBlock()->getParent());
  if (!supportsI8DotDecomposition(rewriter, accElem) || (k % kI8MmaK) != 0)
    return tile;

  // Cap the MMAv2 accumulator at 32 registers per thread.
  int64_t maxTileArea = 32 * 32 * numWarps / (accElem.getWidth() == 64 ? 2 : 1);
  for (int64_t tileM = kI8MmaM; tileM <= m; tileM *= 2) {
    if ((m % tileM) != 0)
      continue;
    for (int64_t tileN = kI8MmaN; tileN <= n; tileN *= 2) {
      if ((n % tileN) == 0 && tileM <= 2 * tileN && tileN <= 2 * tileM &&
          canUseI8MmaTile(tileM, tileN, numWarps) &&
          tileM * tileN <= maxTileArea &&
          tileM * tileN > tile.first * tile.second)
        tile = {tileM, tileN};
    }
  }
  return tile;
}

Operation *createGlobalScratchBarrier(PatternRewriter &rewriter, Location loc,
                                      bool sharedClusterState = false) {
  Operation *barrier = ttg::BarrierOp::create(rewriter, loc,
                                              ttg::AddrSpace::GlobalRead |
                                                  ttg::AddrSpace::GlobalWrite)
                           .getOperation();
  if (sharedClusterState)
    barrier = ttng::ClusterBarrierOp::create(rewriter, loc).getOperation();
  return barrier;
}

void createSynchronousCompletionArrive(PatternRewriter &rewriter, Location loc,
                                       Value barrier, Value pred) {
  // Hardware two-CTA tcgen05 completion is issued by the lead CTA and
  // multicast to its partner. Since FPSAN erases that instruction, each CTA
  // performs the corresponding arrival on its local barrier copy.
  ttng::ArriveBarrierOp::create(rewriter, loc, barrier, 1, pred);
}

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
  ttg::MemDescType scaleSourceType;
};

struct MmaOperandSource {
  Value scratchPtr;
  Value sharedMemdesc;
  RankedTensorType tileType;
  int64_t rowStride;
  int64_t stride;

  bool isShared() const { return static_cast<bool>(sharedMemdesc); }
};

struct ScratchState {
  std::optional<ScratchInfo> canonical;
  DenseMap<Region *, ScratchInfo> byScope;
};

Type getScratchStorageElementType(Type elemTy) {
  if (auto floatTy = dyn_cast<FloatType>(elemTy))
    return IntegerType::get(elemTy.getContext(), floatTy.getWidth());
  return elemTy;
}

RankedTensorType getScratchStorageType(RankedTensorType tensorTy) {
  auto elemTy = getScratchStorageElementType(tensorTy.getElementType());
  return tensorTy.clone(elemTy);
}

Value embedToInt(PatternRewriter &rewriter, Location loc, Value v) {
  if (isa<IntegerType>(getElementTypeOrSelf(v.getType())))
    return v;
  return ExperimentalFPSanEmbedOp::create(rewriter, loc,
                                          getIntTypeLike(v.getType()), v);
}

Value unembedToFloat(PatternRewriter &rewriter, Location loc, Value v,
                     Type floatTy) {
  return ExperimentalFPSanUnembedOp::create(rewriter, loc, floatTy, v);
}

Value loadFpSanScratchMemory(PatternRewriter &rewriter, Location loc,
                             Value alloc, RankedTensorType tensorTy) {
  auto storageTy = getScratchStorageType(tensorTy);
  Value stored = createLoadScratchMemory(rewriter, loc, alloc, storageTy);
  if (isFloatLike(tensorTy))
    return unembedToFloat(rewriter, loc, stored, tensorTy);
  return stored;
}

Operation *storeFpSanScratchMemory(PatternRewriter &rewriter, Location loc,
                                   Value alloc, Value tensor,
                                   RankedTensorType tensorTy) {
  auto storageTy = getScratchStorageType(tensorTy);
  Value stored = tensor;
  if (isFloatLike(tensorTy))
    stored = embedToInt(rewriter, loc, tensor);
  return createStoreScratchMemory(rewriter, loc, alloc, stored, storageTy);
}

class TmemScratchManager {
public:
  explicit TmemScratchManager(bool sharedClusterState)
      : sharedClusterState(sharedClusterState) {}

  bool usesSharedClusterState() const { return sharedClusterState; }

  ttg::GlobalScratchAllocOp createScratchAlloc(PatternRewriter &rewriter,
                                               Location loc, Type ptrType,
                                               int64_t sizeInBytes,
                                               int64_t alignment) {
    return createThirdPartyScratchAlloc(rewriter, loc, ptrType, sizeInBytes,
                                        alignment, sharedClusterState);
  }

  ttg::BlockedEncodingAttr getScratchEncoding(PatternRewriter &rewriter,
                                              Value memdesc,
                                              ttg::MemDescType memTy) {
    auto layout = getOptimizedBlockedEncoding(rewriter, memTy.getShape(),
                                              memTy.getElementType());
    auto cgaLayout = ttg::getCGALayout(memTy.getEncoding());
    if (cgaLayout.getRank() != memTy.getRank()) {
      assert(cgaLayout.getRank() + 1 == memTy.getRank() &&
             "expected at most one extra multibuffering dimension");

      // Ignore the leading pipelining dim when forwarding the CGA layout.
      SmallVector<int64_t> cgaShape = {1};
      llvm::append_range(cgaShape,
                         cgaLayout.getLinearLayout().getOutDimSizes());
      cgaLayout = ttg::CGAEncodingAttr::get(
          rewriter.getContext(),
          cgaLayout.getLinearLayout().reshapeOuts(
              standardOutDimPairs(rewriter.getContext(), cgaShape)));
    }
    return ttg::BlockedEncodingAttr::get(
        rewriter.getContext(), layout.getSizePerThread(),
        layout.getThreadsPerWarp(), layout.getWarpsPerCTA(), layout.getOrder(),
        cgaLayout);
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
      ScratchState &state = scratchMap[memdesc];
      auto itRegion = state.byScope.find(scope);
      if (itRegion != state.byScope.end()) {
        if (itRegion->second.ptr && itRegion->second.ptr.getType())
          return itRegion->second;
        state.byScope.erase(itRegion);
      }
      if (state.canonical) {
        Value ptr =
            remapToScope(state.canonical->ptr, rewriter, scope, alloc.getLoc());
        ScratchInfo info{ptr, state.canonical->tensorType};
        state.byScope[scope] = info;
        return info;
      }

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(alloc);
      auto loc = alloc.getLoc();
      auto layout = getScratchEncoding(rewriter, memdesc, memTy);
      auto tensorTy = RankedTensorType::get(memTy.getShape(),
                                            memTy.getElementType(), layout);
      auto storageElemTy = getScratchStorageElementType(memTy.getElementType());

      int64_t elSize = memTy.getElementType().getIntOrFloatBitWidth() / 8;
      int64_t alignment = std::max<int64_t>(elSize, 16);
      int64_t sizeInBytes = product(memTy.getShape()) * elSize;
      auto ptrTy = triton::getPointerType(storageElemTy);
      auto allocOp =
          createScratchAlloc(rewriter, loc, ptrTy, sizeInBytes, alignment);
      Value ptr = allocOp.getResult();

      if (Value init = alloc.getSrc()) {
        auto initTy = cast<RankedTensorType>(init.getType());
        if (!storeFpSanScratchMemory(rewriter, loc, ptr, init, initTy))
          return std::nullopt;
        createGlobalScratchBarrier(rewriter, loc, sharedClusterState);
      }

      state.canonical = ScratchInfo{ptr, tensorTy};

      ptr = remapToScope(ptr, rewriter, scope, loc);
      ScratchInfo info{ptr, tensorTy};
      state.byScope[scope] = info;
      return info;
    }

    if (auto subslice = memdesc.getDefiningOp<ttng::TMEMSubSliceOp>()) {
      auto baseInfo = getOrCreate(subslice.getSrc(), rewriter, scope);
      if (!baseInfo || baseInfo->scaleSourceType)
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
      if (!baseInfo || baseInfo->scaleSourceType)
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

      auto baseTy = cast<ttg::MemDescType>(view.getSrc().getType());
      if (isa<ttng::TensorMemoryScalesEncodingAttr>(baseTy.getEncoding()) &&
          !isa<ttng::TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
        return ScratchInfo{baseInfo->ptr, baseInfo->tensorType, baseTy};
      if (baseInfo->scaleSourceType) {
        if (isa<ttng::TensorMemoryScalesEncodingAttr>(memTy.getEncoding()))
          return std::nullopt;
        return baseInfo;
      }

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(view);
      auto loc = view.getLoc();
      Value ptr = baseInfo->ptr;
      auto ptrTy = triton::getPointerType(
          getScratchStorageElementType(memTy.getElementType()));
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

  DenseMap<Value, ScratchState> scratchMap;
  bool sharedClusterState;
};

Value createScratchAndStore(PatternRewriter &rewriter, Location loc, Value val,
                            RankedTensorType tensorTy) {
  auto storageTy = getScratchStorageType(tensorTy);
  int64_t elSize = tensorTy.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t alignment = std::max<int64_t>(elSize, 16);
  int64_t sizeInBytes = product(tensorTy.getShape()) * elSize;
  auto ptrTy = triton::getPointerType(storageTy.getElementType());
  auto allocOp = createThirdPartyScratchAlloc(rewriter, loc, ptrTy, sizeInBytes,
                                              alignment);
  if (!storeFpSanScratchMemory(rewriter, loc, allocOp.getResult(), val,
                               tensorTy))
    return Value();
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

LogicalResult emitFpSanUnsupported(Operation *op) {
  op->emitOpError() << "unsupported by fpsan";
  return failure();
}

LogicalResult emitFpSanCodegenError(Operation *op) {
  op->emitOpError() << "fpsan codegen error";
  return failure();
}

LogicalResult emitFpSanInvariantError(Operation *op) {
  assert(false && "unexpected invalid IR in FpSanitizer");
  op->emitOpError() << "fpsan invariant violation";
  return failure();
}

Type getIntTypeLike(Type ty) {
  auto elem = dyn_cast<FloatType>(getElementTypeOrSelf(ty));
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
  auto elem = cast<IntegerType>(getElementTypeOrSelf(ty));
  return elem.getWidth();
}

Type getTypeWithElement(Type ty, Type elemTy) {
  return cast<RankedTensorType>(ty).clone(elemTy);
}

Value getIntConstantLike(PatternRewriter &rewriter, Location loc, Type targetTy,
                         int64_t value) {
  if (auto shaped = dyn_cast<ShapedType>(targetTy)) {
    auto elem = cast<IntegerType>(shaped.getElementType());
    auto intAttr = IntegerAttr::get(
        elem, APInt(elem.getWidth(), static_cast<uint64_t>(value),
                    /*isSigned=*/true, /*implicitTrunc=*/true));
    auto attr = DenseElementsAttr::get(shaped, intAttr);
    return arith::ConstantOp::create(rewriter, loc, attr);
  }
  auto intTy = cast<IntegerType>(targetTy);
  auto attr = IntegerAttr::get(
      intTy, APInt(intTy.getWidth(), static_cast<uint64_t>(value),
                   /*isSigned=*/true, /*implicitTrunc=*/true));
  return arith::ConstantOp::create(rewriter, loc, attr);
}

Value getUIntConstantLike(PatternRewriter &rewriter, Location loc,
                          Type targetTy, uint64_t value) {
  if (auto shaped = dyn_cast<ShapedType>(targetTy)) {
    auto elem = cast<IntegerType>(shaped.getElementType());
    auto intAttr = IntegerAttr::get(elem, APInt(elem.getWidth(), value,
                                                /*isSigned=*/false,
                                                /*implicitTrunc=*/true));
    auto attr = DenseElementsAttr::get(shaped, intAttr);
    return arith::ConstantOp::create(rewriter, loc, attr);
  }
  auto intTy = cast<IntegerType>(targetTy);
  auto attr = IntegerAttr::get(intTy, APInt(intTy.getWidth(), value,
                                            /*isSigned=*/false,
                                            /*implicitTrunc=*/true));
  return arith::ConstantOp::create(rewriter, loc, attr);
}

Value getU32ConstantLike(PatternRewriter &rewriter, Location loc, Type targetTy,
                         uint32_t value) {
  return getUIntConstantLike(rewriter, loc, targetTy, value);
}

Value castSignedIntValueToType(PatternRewriter &rewriter, Location loc, Value v,
                               Type targetTy) {
  if (v.getType() == targetTy)
    return v;

  unsigned srcWidth = getIntBitwidth(v.getType());
  unsigned dstWidth = getIntBitwidth(targetTy);
  if (dstWidth > srcWidth) {
    return arith::ExtSIOp::create(rewriter, loc, targetTy, v);
  }
  if (srcWidth > dstWidth) {
    return arith::TruncIOp::create(rewriter, loc, targetTy, v);
  }
  return v;
}

Value castScalarIntToIntLike(PatternRewriter &rewriter, Location loc,
                             Value scalar, Type targetTy) {
  auto elemTy = cast<IntegerType>(getElementTypeOrSelf(targetTy));
  if (scalar.getType() != elemTy)
    scalar = castSignedIntValueToType(rewriter, loc, scalar, elemTy);
  if (isa<ShapedType>(targetTy))
    return tt::SplatOp::create(rewriter, loc, targetTy, scalar);
  return scalar;
}

uint64_t getLowBitsMask(unsigned bitWidth) {
  assert(bitWidth > 0 && bitWidth <= 64);
  if (bitWidth == 64)
    return ~uint64_t{0};
  return (uint64_t{1} << bitWidth) - 1;
}

uint64_t invOddU64(uint64_t a) {
  assert((a & 1) == 1);
  uint64_t x = 2 - a;
  for (unsigned correctBits = 2; correctBits < 64; correctBits *= 2)
    x *= 2 - a * x;
  return x;
}

Value embedFloatBitsToInt(PatternRewriter &rewriter, Location loc,
                          Value rawBits, FloatType floatElemTy) {
  Type floatTy = floatElemTy;
  if (auto ranked = dyn_cast<RankedTensorType>(rawBits.getType()))
    floatTy = ranked.clone(floatElemTy);
  Value rawFloat = tt::BitcastOp::create(rewriter, loc, floatTy, rawBits);
  return embedToInt(rewriter, loc, rawFloat);
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

constexpr uint32_t kUnaryTagMultiplier = 314159u;

Value fpsanUnaryTagged(PatternRewriter &rewriter, Location loc, Value input,
                       UnaryOpId opId) {
  auto inI = embedToInt(rewriter, loc, input);
  uint64_t opIdHash = murmur64Mixer(getUnaryOpId(opId));
  auto opIdVal = getIntConstantLike(rewriter, loc, inI.getType(),
                                    static_cast<int64_t>(opIdHash));
  auto multiplier =
      getU32ConstantLike(rewriter, loc, inI.getType(), kUnaryTagMultiplier);
  auto mixedIn = arith::MulIOp::create(rewriter, loc, inI, multiplier);
  auto tagged = arith::XOrIOp::create(rewriter, loc, mixedIn, opIdVal);
  auto outI = arith::MulIOp::create(rewriter, loc, tagged, multiplier);
  return unembedToFloat(rewriter, loc, outI, input.getType());
}

Value fpsanIntInv(PatternRewriter &rewriter, Location loc, Value u) {
  auto one = getU32ConstantLike(rewriter, loc, u.getType(), 1u);
  auto two = getU32ConstantLike(rewriter, loc, u.getType(), 2u);
  auto evenMask = getIntConstantLike(rewriter, loc, u.getType(), -2);

  Value a = arith::OrIOp::create(rewriter, loc, u, one);
  Value x = arith::SubIOp::create(rewriter, loc, two, a);
  for (unsigned correctBits = 2; correctBits < getIntBitwidth(u.getType());
       correctBits *= 2) {
    Value ax = arith::MulIOp::create(rewriter, loc, a, x);
    Value factor = arith::SubIOp::create(rewriter, loc, two, ax);
    x = arith::MulIOp::create(rewriter, loc, x, factor);
  }

  Value evenPart = arith::AndIOp::create(rewriter, loc, x, evenMask);
  Value originalParity = arith::AndIOp::create(rewriter, loc, u, one);
  return arith::OrIOp::create(rewriter, loc, evenPart, originalParity);
}

Value fpsanFDiv(PatternRewriter &rewriter, Location loc, Value num, Value den) {
  auto numI = embedToInt(rewriter, loc, num);
  auto denI = embedToInt(rewriter, loc, den);
  auto inv = fpsanIntInv(rewriter, loc, denI);
  auto resI = arith::MulIOp::create(rewriter, loc, numI, inv);
  return unembedToFloat(rewriter, loc, resI, num.getType());
}

Value fpsanFma(PatternRewriter &rewriter, Location loc, Value a, Value b,
               Value c) {
  auto aI = embedToInt(rewriter, loc, a);
  auto bI = embedToInt(rewriter, loc, b);
  auto cI = embedToInt(rewriter, loc, c);
  auto mul = arith::MulIOp::create(rewriter, loc, aI, bI);
  auto sum = arith::AddIOp::create(rewriter, loc, mul, cI);
  return unembedToFloat(rewriter, loc, sum, a.getType());
}

Value fpsanSRem(PatternRewriter &rewriter, Location loc, Value num, Value den) {
  auto numI = embedToInt(rewriter, loc, num);
  auto denI = embedToInt(rewriter, loc, den);
  auto one = getIntConstantLike(rewriter, loc, denI.getType(), 1);
  auto denSafe = arith::OrIOp::create(rewriter, loc, denI, one);
  auto resI = arith::RemSIOp::create(rewriter, loc, numI, denSafe);
  return unembedToFloat(rewriter, loc, resI, num.getType());
}

// Modular exponentiation in payload space; this preserves
// exp2(a + b) = exp2(a) * exp2(b) under the integer rewrite.
Value fpsanExp2FromInt(PatternRewriter &rewriter, Location loc, Value xI,
                       Type floatTy) {
  unsigned bitWidth = getIntBitwidth(xI.getType());
  auto one = getIntConstantLike(rewriter, loc, xI.getType(), 1);
  auto zero = getIntConstantLike(rewriter, loc, xI.getType(), 0);
  auto c = getIntConstantLike(rewriter, loc, xI.getType(), 0xa343836d);

  auto lower =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  auto upper = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getI32IntegerAttr(bitWidth));
  auto step =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
  auto topBit = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(bitWidth - 1));
  auto loop = scf::ForOp::create(rewriter, loc, lower, upper, step, one);
  rewriter.setInsertionPointToStart(loop.getBody());

  Value i = loop.getInductionVar();
  Value y = loop.getRegionIterArgs()[0];
  y = arith::MulIOp::create(rewriter, loc, y, y);
  Value bitIndex =
      arith::SubIOp::create(rewriter, loc, rewriter.getI32Type(), topBit, i);
  Value shift = castScalarIntToIntLike(rewriter, loc, bitIndex, xI.getType());
  Value bit = arith::ShLIOp::create(rewriter, loc, one, shift);
  auto masked = arith::AndIOp::create(rewriter, loc, xI, bit);
  auto isZero = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      masked, zero);
  auto factor = arith::SelectOp::create(rewriter, loc, isZero, one, c);
  y = arith::MulIOp::create(rewriter, loc, y, factor);
  scf::YieldOp::create(rewriter, loc, y);
  rewriter.setInsertionPointAfter(loop);

  return unembedToFloat(rewriter, loc, loop.getResult(0), floatTy);
}

Value fpsanExp2(PatternRewriter &rewriter, Location loc, Value input) {
  auto elemTy = dyn_cast<FloatType>(getElementTypeOrSelf(input.getType()));
  if (!elemTy)
    return Value();
  return fpsanExp2FromInt(rewriter, loc, embedToInt(rewriter, loc, input),
                          input.getType());
}

Value fpsanExp(PatternRewriter &rewriter, Location loc, Value input) {
  auto elemTy = dyn_cast<FloatType>(getElementTypeOrSelf(input.getType()));
  if (!elemTy)
    return Value();

  auto inputI = embedToInt(rewriter, loc, input);
  auto rcpLog2 =
      getU32ConstantLike(rewriter, loc, inputI.getType(), 0x236ee9bfu);
  auto scaledI = arith::MulIOp::create(rewriter, loc, inputI, rcpLog2);
  return fpsanExp2FromInt(rewriter, loc, scaledI, input.getType());
}

struct FpSanCosSin {
  Value cos;
  Value sin;
};

FpSanCosSin fpsanCosSinPayload(PatternRewriter &rewriter, Location loc,
                               Value xI) {
  Type intTy = xI.getType();
  unsigned bitWidth = getIntBitwidth(intTy);
  uint64_t mask = getLowBitsMask(bitWidth);
  uint64_t rcp5 = invOddU64(5) & mask;
  uint64_t aValue = (uint64_t{0} - ((uint64_t{3} * rcp5) & mask)) & mask;
  uint64_t bValue = (uint64_t{4} * rcp5) & mask;

  auto zero = getUIntConstantLike(rewriter, loc, intTy, 0);
  auto one = getUIntConstantLike(rewriter, loc, intTy, 1);
  auto two = getUIntConstantLike(rewriter, loc, intTy, 2);
  auto a = getUIntConstantLike(rewriter, loc, intTy, aValue);
  auto b = getUIntConstantLike(rewriter, loc, intTy, bValue);

  auto lower =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  auto upper = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getI32IntegerAttr(bitWidth));
  auto step =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
  auto topBit = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(bitWidth - 1));
  SmallVector<Value> initArgs{one, zero};
  auto loop = scf::ForOp::create(rewriter, loc, lower, upper, step, initArgs);
  rewriter.setInsertionPointToStart(loop.getBody());

  Value bit = loop.getInductionVar();
  Value c = loop.getRegionIterArgs()[0];
  Value s = loop.getRegionIterArgs()[1];
  Value cc = arith::MulIOp::create(rewriter, loc, c, c);
  Value ss = arith::MulIOp::create(rewriter, loc, s, s);
  Value cDouble = arith::SubIOp::create(rewriter, loc, cc, ss);
  Value cs = arith::MulIOp::create(rewriter, loc, c, s);
  Value sDouble = arith::MulIOp::create(rewriter, loc, two, cs);

  Value ac = arith::MulIOp::create(rewriter, loc, a, cDouble);
  Value bs = arith::MulIOp::create(rewriter, loc, b, sDouble);
  Value cInc = arith::SubIOp::create(rewriter, loc, ac, bs);
  Value as = arith::MulIOp::create(rewriter, loc, a, sDouble);
  Value bc = arith::MulIOp::create(rewriter, loc, b, cDouble);
  Value sInc = arith::AddIOp::create(rewriter, loc, as, bc);

  Value bitIndex =
      arith::SubIOp::create(rewriter, loc, rewriter.getI32Type(), topBit, bit);
  Value shift = castScalarIntToIntLike(rewriter, loc, bitIndex, intTy);
  Value bitMask = arith::ShLIOp::create(rewriter, loc, one, shift);
  auto masked = arith::AndIOp::create(rewriter, loc, xI, bitMask);
  auto isZero = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq,
                                      masked, zero);
  c = arith::SelectOp::create(rewriter, loc, isZero, cDouble, cInc);
  s = arith::SelectOp::create(rewriter, loc, isZero, sDouble, sInc);
  scf::YieldOp::create(rewriter, loc, ValueRange{c, s});
  rewriter.setInsertionPointAfter(loop);

  return {loop.getResult(0), loop.getResult(1)};
}

Value fpsanCos(PatternRewriter &rewriter, Location loc, Value input) {
  if (!isFloatLike(input.getType()))
    return Value();
  auto cosSin =
      fpsanCosSinPayload(rewriter, loc, embedToInt(rewriter, loc, input));
  return unembedToFloat(rewriter, loc, cosSin.cos, input.getType());
}

Value fpsanSin(PatternRewriter &rewriter, Location loc, Value input) {
  if (!isFloatLike(input.getType()))
    return Value();
  auto cosSin =
      fpsanCosSinPayload(rewriter, loc, embedToInt(rewriter, loc, input));
  return unembedToFloat(rewriter, loc, cosSin.sin, input.getType());
}

bool externHasNumericOperands(tt::ExternElementwiseOp op) {
  return llvm::all_of(op.getOperands(), [](Value operand) {
    return isNumericLike(operand.getType());
  });
}

Value castExternOperandToResultInt(PatternRewriter &rewriter, Location loc,
                                   Value operand, Type resultIntTy) {
  if (isFloatLike(operand.getType())) {
    return castSignedIntValueToType(
        rewriter, loc, embedToInt(rewriter, loc, operand), resultIntTy);
  }
  if (isIntLike(operand.getType())) {
    return castSignedIntValueToType(rewriter, loc, operand, resultIntTy);
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
  return unembedToFloat(rewriter, loc, outI, resultTy);
}

std::optional<ScratchInfo>
createTmemOperandScratch(PatternRewriter &rewriter, Location loc,
                         TmemScratchManager &scratch, Value memdesc,
                         ttg::MemDescType memTy, Region *scope) {
  auto layout = scratch.getScratchEncoding(rewriter, memdesc, memTy);
  auto tensorTy =
      RankedTensorType::get(memTy.getShape(), memTy.getElementType(), layout);
  auto info = scratch.getOrCreate(memdesc, rewriter, scope);
  if (!info || info->scaleSourceType)
    return std::nullopt;
  Value fullVal = loadFpSanScratchMemory(rewriter, loc, info->ptr, tensorTy);
  if (!fullVal)
    return std::nullopt;
  int64_t elSize = memTy.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t alignment = std::max<int64_t>(elSize, 16);
  int64_t sizeInBytes = product(memTy.getShape()) * elSize;
  auto ptrTy = triton::getPointerType(
      getScratchStorageElementType(memTy.getElementType()));
  auto allocOp =
      scratch.createScratchAlloc(rewriter, loc, ptrTy, sizeInBytes, alignment);
  Value ptr = allocOp.getResult();
  if (!storeFpSanScratchMemory(rewriter, loc, ptr, fullVal, tensorTy))
    return std::nullopt;
  return ScratchInfo{ptr, tensorTy};
}

std::optional<MmaOperandSource> createMmaOperandSource(
    PatternRewriter &rewriter, Location loc, TmemScratchManager &scratch,
    Value memdesc, ttg::MemDescType memTy, bool isTmem, RankedTensorType tileTy,
    Region *scope, int64_t rowStride, int64_t stride) {
  if (!isTmem)
    return MmaOperandSource{Value(), memdesc, tileTy, rowStride, stride};
  auto info =
      createTmemOperandScratch(rewriter, loc, scratch, memdesc, memTy, scope);
  if (!info)
    return std::nullopt;
  return MmaOperandSource{info->ptr, Value(), tileTy, rowStride, stride};
}

std::optional<ScratchInfo> createWGMMAScratch(PatternRewriter &rewriter,
                                              Location loc, Value operand) {
  if (auto memTy = dyn_cast<ttg::MemDescType>(operand.getType())) {
    auto layout = getOptimizedBlockedEncoding(rewriter, memTy.getShape(),
                                              memTy.getElementType());
    auto tensorTy =
        RankedTensorType::get(memTy.getShape(), memTy.getElementType(), layout);
    Value fullVal =
        ttg::LocalLoadOp::create(rewriter, loc, tensorTy, operand, Value())
            .getResult();
    Value ptr = createScratchAndStore(rewriter, loc, fullVal, tensorTy);
    if (!ptr)
      return std::nullopt;
    return ScratchInfo{ptr, tensorTy};
  }

  auto tensorTy = dyn_cast<RankedTensorType>(operand.getType());
  if (!tensorTy)
    return std::nullopt;
  Value ptr = createScratchAndStore(rewriter, loc, operand, tensorTy);
  if (!ptr)
    return std::nullopt;
  return ScratchInfo{ptr, tensorTy};
}

Value createAsyncToken(PatternRewriter &rewriter, Location loc,
                       ValueRange deps) {
  return ttg::AsyncCommitGroupOp::create(rewriter, loc, deps).getResult();
}

Value createPointerTensorStrided2D(PatternRewriter &rewriter, Location loc,
                                   Value base, RankedTensorType resultTy,
                                   int64_t stride0, int64_t stride1) {
  auto shape = resultTy.getShape();
  auto encoding = cast<ttg::DistributedEncodingTrait>(resultTy.getEncoding());
  auto ptrTy = base.getType();
  auto ptrTensorTy = RankedTensorType::get(shape, ptrTy, encoding);
  Value ptrTensor = tt::SplatOp::create(rewriter, loc, ptrTensorTy, base);
  auto i32Ty = rewriter.getI32Type();
  auto offsetsTy = RankedTensorType::get(shape, i32Ty, encoding);

  auto dim0Ty = getSlicedTensorType(offsetsTy, {0}, i32Ty);
  auto range0 = tt::MakeRangeOp::create(rewriter, loc, dim0Ty, 0, shape[0]);
  auto stride0Const = createConstIntTensor(rewriter, loc, stride0, dim0Ty);
  auto off0 =
      arith::MulIOp::create(rewriter, loc, dim0Ty, range0, stride0Const);
  auto off0Exp = reshapeAndBroadcast(rewriter, loc, off0, {0}, offsetsTy);
  ptrTensor =
      tt::AddPtrOp::create(rewriter, loc, ptrTensorTy, ptrTensor, off0Exp);

  auto dim1Ty = getSlicedTensorType(offsetsTy, {1}, i32Ty);
  auto range1 = tt::MakeRangeOp::create(rewriter, loc, dim1Ty, 0, shape[1]);
  auto stride1Const = createConstIntTensor(rewriter, loc, stride1, dim1Ty);
  auto off1 =
      arith::MulIOp::create(rewriter, loc, dim1Ty, range1, stride1Const);
  auto off1Exp = reshapeAndBroadcast(rewriter, loc, off1, {1}, offsetsTy);
  ptrTensor =
      tt::AddPtrOp::create(rewriter, loc, ptrTensorTy, ptrTensor, off1Exp);

  return ptrTensor;
}

Value loadScratchStrided2D(PatternRewriter &rewriter, Location loc, Value base,
                           RankedTensorType tensorTy, int64_t stride0,
                           int64_t stride1) {
  auto storageTy = getScratchStorageType(tensorTy);
  auto ptrTensor = createPointerTensorStrided2D(rewriter, loc, base, storageTy,
                                                stride0, stride1);
  Value stored =
      tt::LoadOp::create(rewriter, loc, ptrTensor, CacheModifier::NONE,
                         EvictionPolicy::NORMAL, false);
  if (isFloatLike(tensorTy))
    return unembedToFloat(rewriter, loc, stored, tensorTy);
  return stored;
}

Value loadScratchStrided2D(PatternRewriter &rewriter, Location loc, Value base,
                           RankedTensorType tensorTy, int64_t stride1) {
  return loadScratchStrided2D(rewriter, loc, base, tensorTy, /*stride0=*/1,
                              stride1);
}

Value loadMmaOperand(PatternRewriter &rewriter, Location loc,
                     const MmaOperandSource &source, RankedTensorType resultTy,
                     bool isLhs, Value tileOffset, Value kOffset) {
  if (!source.isShared()) {
    Value rowOffset = isLhs ? tileOffset : kOffset;
    Value colOffset = isLhs ? kOffset : tileOffset;
    Value rowStride = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(source.rowStride));
    Value stride = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(source.stride));
    Value row = arith::MulIOp::create(rewriter, loc, rowOffset, rowStride);
    Value col = arith::MulIOp::create(rewriter, loc, colOffset, stride);
    Value offset = arith::AddIOp::create(rewriter, loc, row, col);
    Value ptr = tt::AddPtrOp::create(rewriter, loc, source.scratchPtr.getType(),
                                     source.scratchPtr, offset);
    return loadScratchStrided2D(rewriter, loc, ptr, resultTy, source.rowStride,
                                source.stride);
  }

  Value shared = source.sharedMemdesc;
  unsigned tileAxis = isLhs ? 0 : 1;
  unsigned kAxis = 1 - tileAxis;
  ArrayRef<int64_t> loadShape = resultTy.getShape();
  auto indicesTy = resultTy.clone(rewriter.getI32Type());
  auto kTy = getSlicedTensorType(resultTy, {static_cast<int>(kAxis)},
                                 rewriter.getI32Type());
  Value indices =
      tt::MakeRangeOp::create(rewriter, loc, kTy, 0, loadShape[kAxis]);
  Value kSplat = tt::SplatOp::create(rewriter, loc, kTy, kOffset);
  indices = arith::AddIOp::create(rewriter, loc, indices, kSplat);
  indices = reshapeAndBroadcast(rewriter, loc, indices,
                                {static_cast<int>(kAxis)}, indicesTy);
  Value zero =
      arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
  SmallVector<Value> offsets(2, zero);
  offsets[tileAxis] = tileOffset;
  return ExperimentalLocalGatherOp::create(rewriter, loc, resultTy, shared,
                                           indices, offsets,
                                           rewriter.getI32IntegerAttr(kAxis));
}

Operation *storeScratchStrided2D(PatternRewriter &rewriter, Location loc,
                                 Value base, Value tensor,
                                 RankedTensorType tensorTy, int64_t stride0,
                                 int64_t stride1) {
  auto storageTy = getScratchStorageType(tensorTy);
  auto ptrTensor = createPointerTensorStrided2D(rewriter, loc, base, storageTy,
                                                stride0, stride1);
  Value stored = tensor;
  if (isFloatLike(tensorTy))
    stored = embedToInt(rewriter, loc, tensor);
  return tt::StoreOp::create(rewriter, loc, ptrTensor, stored,
                             CacheModifier::NONE, EvictionPolicy::NORMAL);
}

Value unpackPackedFp4Slice(PatternRewriter &rewriter, Location loc,
                           Value packedSlice, Value kI32) {
  Value packedI = embedToInt(rewriter, loc, packedSlice);
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

FloatType getDotScaledComputeFloatType(PatternRewriter &rewriter,
                                       tt::ScaleDotElemType aElemType,
                                       tt::ScaleDotElemType bElemType) {
  if (aElemType == tt::ScaleDotElemType::FP16 ||
      bElemType == tt::ScaleDotElemType::FP16)
    return Float16Type::get(rewriter.getContext());
  return BFloat16Type::get(rewriter.getContext());
}

FloatType getDotScaledStorageFloatType(PatternRewriter &rewriter,
                                       tt::ScaleDotElemType elemType) {
  MLIRContext *ctx = rewriter.getContext();
  switch (elemType) {
  case tt::ScaleDotElemType::E4M3:
    return Float8E4M3FNType::get(ctx);
  case tt::ScaleDotElemType::E5M2:
    return Float8E5M2Type::get(ctx);
  case tt::ScaleDotElemType::BF16:
    return BFloat16Type::get(ctx);
  case tt::ScaleDotElemType::FP16:
    return Float16Type::get(ctx);
  default:
    return {};
  }
}

Value castDotScaledOperandToComputePayload(PatternRewriter &rewriter,
                                           Location loc, Value slice,
                                           tt::ScaleDotElemType elemType,
                                           FloatType computeElem) {
  Type computeIntTy = getTypeWithElement(
      slice.getType(), IntegerType::get(rewriter.getContext(),
                                        computeElem.getIntOrFloatBitWidth()));

  if (auto storageFloat = getDotScaledStorageFloatType(rewriter, elemType)) {
    Value payload;
    if (isFloatLike(slice.getType())) {
      payload = embedToInt(rewriter, loc, slice);
    } else {
      Value raw = castSignedIntValueToType(
          rewriter, loc, slice,
          getTypeWithElement(slice.getType(),
                             IntegerType::get(rewriter.getContext(),
                                              storageFloat.getWidth())));
      payload = embedFloatBitsToInt(rewriter, loc, raw, storageFloat);
    }
    return castSignedIntValueToType(rewriter, loc, payload, computeIntTy);
  }

  // Match ttg.fp4_to_fp sanitization: unpacked e2m1 nibbles are payloads in
  // the destination floating type.  The 6-bit formats are not packed here, but
  // use the same payload-preserving integer cast until we add a float6 mixer.
  Value rawPayload = embedToInt(rewriter, loc, slice);
  return castSignedIntValueToType(rewriter, loc, rawPayload, computeIntTy);
}

Value scaleI8ToF32Payload(PatternRewriter &rewriter, Location loc,
                          Value scaleI) {
  auto i32Elem = rewriter.getI32Type();
  auto i32Ty = getTypeWithElement(scaleI.getType(), i32Elem);
  Value scaleI32 = arith::ExtUIOp::create(rewriter, loc, i32Ty, scaleI);
  auto shift = getUIntConstantLike(rewriter, loc, i32Ty, 23);
  Value rawF32 = arith::ShLIOp::create(rewriter, loc, scaleI32, shift);
  return embedFloatBitsToInt(rewriter, loc, rawF32, rewriter.getF32Type());
}

Value scaleI8ToComputePayload(PatternRewriter &rewriter, Location loc,
                              Value scaleI, FloatType computeElem) {
  unsigned computeWidth = computeElem.getIntOrFloatBitWidth();
  Type computeIntTy = getTypeWithElement(
      scaleI.getType(), IntegerType::get(rewriter.getContext(), computeWidth));

  if (computeElem == rewriter.getF16Type()) {
    // The real decomposition builds an f32 E8M0 scale and truncates it to f16.
    // Under FPSan, truncf means mix-f32, signed-truncate, unmix-f16.
    Value payloadF32 = scaleI8ToF32Payload(rewriter, loc, scaleI);
    return castSignedIntValueToType(rewriter, loc, payloadF32, computeIntTy);
  }

  Value scaleComputeI =
      arith::ExtUIOp::create(rewriter, loc, computeIntTy, scaleI);
  unsigned shiftValue = computeElem.getFPMantissaWidth() - 1;
  auto shift = getUIntConstantLike(rewriter, loc, computeIntTy, shiftValue);
  Value rawCompute = arith::ShLIOp::create(rewriter, loc, scaleComputeI, shift);
  return embedFloatBitsToInt(rewriter, loc, rawCompute, computeElem);
}

Value castDotScaledScaleToComputePayload(PatternRewriter &rewriter,
                                         Location loc, Value scaleSlice,
                                         FloatType computeElem) {
  Type computeIntTy =
      getTypeWithElement(scaleSlice.getType(),
                         IntegerType::get(rewriter.getContext(),
                                          computeElem.getIntOrFloatBitWidth()));
  if (isFloatLike(scaleSlice.getType())) {
    Value payload = embedToInt(rewriter, loc, scaleSlice);
    return castSignedIntValueToType(rewriter, loc, payload, computeIntTy);
  }
  return scaleI8ToComputePayload(
      rewriter, loc, embedToInt(rewriter, loc, scaleSlice), computeElem);
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
  tt::ScaleDotElemType aElemType;
  tt::ScaleDotElemType bElemType;
  FloatType computeElem;
};

Value loadScaleSlice(PatternRewriter &rewriter, Location loc, bool isLhs,
                     const DotScaleConfig &scale, Value tileIdx, Value kI32) {
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
                     IntegerType accElem, const DotScaleConfig &scale = {}) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto fullTy = RankedTensorType::get({m, n}, accElem, accLayout);

  Value aI;
  Value bI;
  if (scale.computeElem) {
    aI = castDotScaledOperandToComputePayload(
        rewriter, loc, aSlice, scale.aElemType, scale.computeElem);
    bI = castDotScaledOperandToComputePayload(
        rewriter, loc, bSlice, scale.bElemType, scale.computeElem);
    if (aScaleSlice) {
      auto aScaleI = castDotScaledScaleToComputePayload(
          rewriter, loc, aScaleSlice, scale.computeElem);
      aI = arith::MulIOp::create(rewriter, loc, aI, aScaleI);
    }
    if (bScaleSlice) {
      auto bScaleI = castDotScaledScaleToComputePayload(
          rewriter, loc, bScaleSlice, scale.computeElem);
      bI = arith::MulIOp::create(rewriter, loc, bI, bScaleI);
    }
  } else {
    aI = embedToInt(rewriter, loc, aSlice);
    bI = embedToInt(rewriter, loc, bSlice);
  }
  aI = castSignedIntValueToType(rewriter, loc, aI,
                                getTypeWithElement(aI.getType(), accElem));
  bI = castSignedIntValueToType(rewriter, loc, bI,
                                getTypeWithElement(bI.getType(), accElem));
  Value aFull = tt::BroadcastOp::create(rewriter, loc, fullTy, aI);
  Value bFull = tt::BroadcastOp::create(rewriter, loc, fullTy, bI);
  return arith::MulIOp::create(rewriter, loc, aFull, bFull);
}

Value unpackPackedFp4Tensor(PatternRewriter &rewriter, Location loc,
                            Value packed, int64_t axis,
                            RankedTensorType logicalTy) {
  Value packedI = embedToInt(rewriter, loc, packed);
  auto packedTy = cast<RankedTensorType>(packedI.getType());
  auto packedI8Ty = packedTy.clone(rewriter.getI8Type());
  packedI = castSignedIntValueToType(rewriter, loc, packedI, packedI8Ty);

  Value mask = getIntConstantLike(rewriter, loc, packedI8Ty, 0x0F);
  Value four = getIntConstantLike(rewriter, loc, packedI8Ty, 4);
  Value lo = arith::AndIOp::create(rewriter, loc, packedI, mask);
  Value hi = arith::ShRUIOp::create(rewriter, loc, packedI, four);
  auto halfTy = packedTy.clone(logicalTy.getElementType());
  lo = castSignedIntValueToType(rewriter, loc, lo, halfTy);
  hi = castSignedIntValueToType(rewriter, loc, hi, halfTy);
  Value joined = tt::JoinOp::create(rewriter, loc, lo, hi);

  int64_t rank = packedTy.getRank();
  auto order = llvm::to_vector(llvm::seq<int32_t>(axis + 1));
  order.push_back(rank);
  llvm::append_range(order, llvm::seq<int32_t>(axis + 1, rank));
  Value transposed = tt::TransOp::create(rewriter, loc, joined, order);

  Value logical =
      tt::ReshapeOp::create(rewriter, loc, logicalTy.getShape(), transposed);
  return ttg::ConvertLayoutOp::create(rewriter, loc, logicalTy, logical);
}

Value loadOperandK32(PatternRewriter &rewriter, Location loc, bool isLhs,
                     const MmaOperandSource &source, Value tileIdx, Value kI32,
                     ttg::NvidiaMmaEncodingAttr mmaLayout,
                     int64_t packFactor = 1) {
  SmallVector<int64_t> logicalShape{
      isLhs ? source.tileType.getShape()[0] : kI8MmaK,
      isLhs ? kI8MmaK : source.tileType.getShape()[1]};
  SmallVector<int64_t> rawShape = logicalShape;
  rawShape[isLhs ? 1 : 0] /= packFactor;
  auto dotLayout = ttg::DotOperandEncodingAttr::get(
      rewriter.getContext(), !isLhs, mmaLayout, rewriter.getI8Type());
  auto rawLayout = ttg::DotOperandEncodingAttr::get(
      rewriter.getContext(), dotLayout.getOpIdx(), dotLayout.getParent(),
      dotLayout.getKWidth() / packFactor);
  auto rawTy = RankedTensorType::get(rawShape, source.tileType.getElementType(),
                                     rawLayout);

  Value packedKIdx = kI32;
  if (packFactor != 1) {
    Value factor = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(packFactor));
    packedKIdx = arith::DivUIOp::create(rewriter, loc, kI32, factor);
  }
  Value chunk =
      loadMmaOperand(rewriter, loc, source, rawTy, isLhs, tileIdx, packedKIdx);

  if (packFactor == 2) {
    auto logicalTy =
        RankedTensorType::get(logicalShape, rewriter.getI8Type(), dotLayout);
    chunk = unpackPackedFp4Tensor(rewriter, loc, chunk,
                                  /*axis=*/isLhs ? 1 : 0, logicalTy);
  }
  return chunk;
}

Value loadScaledScaleK32(PatternRewriter &rewriter, Location loc, bool isLhs,
                         const DotScaleConfig &scale, Value tileIdx, Value kI32,
                         RankedTensorType targetTy) {
  Value ptr = isLhs ? scale.aScalePtr : scale.bScalePtr;
  if (!ptr)
    return Value();

  int64_t scaleFactor = isLhs ? scale.aScaleFactor : scale.bScaleFactor;
  int64_t scaleStride = isLhs ? scale.aScaleStride : scale.bScaleStride;
  auto scaleTileTy = isLhs ? scale.aScaleTileTy : scale.bScaleTileTy;
  int64_t groups = scaleFactor < kI8MmaK ? kI8MmaK / scaleFactor : 1;
  int64_t repeat = kI8MmaK / groups;
  SmallVector<int64_t> compactShape =
      isLhs ? SmallVector<int64_t>{targetTy.getShape()[0], groups}
            : SmallVector<int64_t>{groups, targetTy.getShape()[1]};
  SmallVector<int64_t> broadcastShape =
      isLhs ? SmallVector<int64_t>{targetTy.getShape()[0], groups, repeat}
            : SmallVector<int64_t>{groups, repeat, targetTy.getShape()[1]};
  int64_t expandAxis = isLhs ? 2 : 1;

  auto broadcastLayout = getOptimizedBlockedEncoding(
      rewriter, broadcastShape, scaleTileTy.getElementType());
  auto compactSliceLayout = ttg::SliceEncodingAttr::get(
      rewriter.getContext(), expandAxis, broadcastLayout);
  auto compactLoadLayout = getOptimizedBlockedEncoding(
      rewriter, compactShape, scaleTileTy.getElementType());
  auto compactLoadTy = RankedTensorType::get(
      compactShape, scaleTileTy.getElementType(), compactLoadLayout);

  Value tilePtr =
      tt::AddPtrOp::create(rewriter, loc, ptr.getType(), ptr, tileIdx);
  Value factor = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(scaleFactor));
  Value groupIdx = arith::DivUIOp::create(rewriter, loc, kI32, factor);
  Value stride = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(scaleStride));
  Value groupOffset = arith::MulIOp::create(rewriter, loc, groupIdx, stride);
  Value groupPtr =
      tt::AddPtrOp::create(rewriter, loc, ptr.getType(), tilePtr, groupOffset);
  Value compact = loadScratchStrided2D(rewriter, loc, groupPtr, compactLoadTy,
                                       /*stride0=*/isLhs ? 1 : scaleStride,
                                       /*stride1=*/isLhs ? scaleStride : 1);
  compact = castDotScaledScaleToComputePayload(rewriter, loc, compact,
                                               scale.computeElem);
  auto compactSliceTy = cast<RankedTensorType>(compact.getType())
                            .cloneWithEncoding(compactSliceLayout);
  compact =
      ttg::ConvertLayoutOp::create(rewriter, loc, compactSliceTy, compact);

  Value expanded = tt::ExpandDimsOp::create(rewriter, loc, compact, expandAxis);
  auto broadcastTy =
      cast<RankedTensorType>(expanded.getType()).clone(broadcastShape);
  Value broadcast =
      tt::BroadcastOp::create(rewriter, loc, broadcastTy, expanded);
  Value reshaped =
      tt::ReshapeOp::create(rewriter, loc, targetTy.getShape(), broadcast);
  if (reshaped.getType() != targetTy)
    reshaped = ttg::ConvertLayoutOp::create(rewriter, loc, targetTy, reshaped);
  return reshaped;
}

Value loadScaledOperandK32(PatternRewriter &rewriter, Location loc, bool isLhs,
                           const MmaOperandSource &source,
                           const DotScaleConfig &scale, Value tileIdx,
                           Value kI32, ttg::NvidiaMmaEncodingAttr mmaLayout) {
  int64_t packFactor = isLhs ? scale.aKPackFactor : scale.bKPackFactor;
  tt::ScaleDotElemType elemType = isLhs ? scale.aElemType : scale.bElemType;
  Value chunk = loadOperandK32(rewriter, loc, isLhs, source, tileIdx, kI32,
                               mmaLayout, packFactor);

  Value payload = castDotScaledOperandToComputePayload(
      rewriter, loc, chunk, elemType, scale.computeElem);
  auto payloadTy = cast<RankedTensorType>(payload.getType());
  Value scalePayload =
      loadScaledScaleK32(rewriter, loc, isLhs, scale, tileIdx, kI32, payloadTy);
  if (scalePayload)
    payload = arith::MulIOp::create(rewriter, loc, payload, scalePayload);
  return payload;
}

Value emitI8DotDecomposition(PatternRewriter &rewriter, Location loc,
                             Value aPayload, Value bPayload,
                             IntegerType accElem, Value initialAccumulator) {
  auto aPayloadTy = cast<RankedTensorType>(aPayload.getType());
  auto bPayloadTy = cast<RankedTensorType>(bPayload.getType());
  auto workMmaTy = cast<RankedTensorType>(initialAccumulator.getType());
  assert(aPayloadTy.getRank() == 2 && bPayloadTy.getRank() == 2 &&
         workMmaTy.getRank() == 2);
  auto aShape = aPayloadTy.getShape();
  auto bShape = bPayloadTy.getShape();
  auto workShape = workMmaTy.getShape();
  assert(aShape[1] == bShape[0] && (aShape[1] % kI8MmaK) == 0);
  assert(aShape[0] == workShape[0] && bShape[1] == workShape[1]);
  auto aElem = cast<IntegerType>(aPayloadTy.getElementType());
  auto bElem = cast<IntegerType>(bPayloadTy.getElementType());
  assert((aElem.getWidth() % 8) == 0 && (bElem.getWidth() % 8) == 0);
  int64_t aLimbs = aElem.getWidth() / 8;
  int64_t bLimbs = bElem.getWidth() / 8;
  int64_t accLimbs = accElem.getWidth() / 8;
  int64_t highestDiagonal = std::min(accLimbs - 1, aLimbs + bLimbs - 2);

  auto *ctx = rewriter.getContext();
  auto i8Ty = rewriter.getI8Type();
  auto i32Ty = rewriter.getI32Type();
  auto mmaLayout = cast<ttg::NvidiaMmaEncodingAttr>(workMmaTy.getEncoding());
  auto aDotLayout = ttg::DotOperandEncodingAttr::get(ctx, 0, mmaLayout, i8Ty);
  auto bDotLayout = ttg::DotOperandEncodingAttr::get(ctx, 1, mmaLayout, i8Ty);
  auto aMmaTy = aPayloadTy.cloneWithEncoding(aDotLayout);
  auto bMmaTy = bPayloadTy.cloneWithEncoding(bDotLayout);
  aPayload = ttg::ConvertLayoutOp::create(rewriter, loc, aMmaTy, aPayload);
  bPayload = ttg::ConvertLayoutOp::create(rewriter, loc, bMmaTy, bPayload);
  auto workElem = cast<IntegerType>(workMmaTy.getElementType());
  assert(workElem == (accElem.getWidth() == 64 ? accElem : i32Ty));

  // Peel register repetitions outside each native IMMA fragment from the
  // largest stride down, then reassemble them in the inverse order.
  SmallVector<std::pair<unsigned, int64_t>> fragmentSplits;
  auto mmaLinearLayout = mmaLayout.toLinearLayout(workShape);
  auto kRegister = StringAttr::get(ctx, "register");
  const auto &registerBases = mmaLinearLayout.getBases().lookup(kRegister);
  for (const auto &basis : llvm::reverse(registerBases)) {
    if (basis[0] >= kI8MmaM && basis[1] == 0)
      fragmentSplits.emplace_back(0, basis[0]);
    else if (basis[0] == 0 && basis[1] >= kI8MmaN)
      fragmentSplits.emplace_back(1, basis[1]);
  }

  auto splitAtRegisterBasis = [&](Value tensor, unsigned axis,
                                  int64_t stride) -> std::pair<Value, Value> {
    auto tensorTy = cast<RankedTensorType>(tensor.getType());
    auto shape = llvm::to_vector(tensorTy.getShape());
    assert(shape.size() == 2 && axis < 2 && (shape[axis] % (2 * stride)) == 0);
    SmallVector<int64_t> expandedShape(shape);
    expandedShape[axis] /= 2 * stride;
    expandedShape.insert(expandedShape.begin() + axis + 1, 2);
    expandedShape.insert(expandedShape.begin() + axis + 2, stride);
    int32_t selectorAxis = axis + 1;
    auto order = llvm::to_vector(llvm::seq<int32_t>(expandedShape.size()));
    order.erase(order.begin() + selectorAxis);
    order.push_back(selectorAxis);
    Value expanded =
        tt::ReshapeOp::create(rewriter, loc, expandedShape, tensor);
    Value transposed = tt::TransOp::create(rewriter, loc, expanded, order);
    auto split = tt::SplitOp::create(rewriter, loc, transposed);

    shape[axis] /= 2;
    return {tt::ReshapeOp::create(rewriter, loc, shape, split.getOutLHS()),
            tt::ReshapeOp::create(rewriter, loc, shape, split.getOutRHS())};
  };

  auto joinAtRegisterBasis = [&](Value lhs, Value rhs, unsigned axis,
                                 int64_t stride) -> Value {
    auto halfTy = cast<RankedTensorType>(lhs.getType());
    auto fullShape = llvm::to_vector(halfTy.getShape());
    assert(fullShape.size() == 2 && axis < 2);
    fullShape[axis] *= 2;
    SmallVector<int64_t> expandedHalfShape(fullShape);
    expandedHalfShape[axis] /= 2 * stride;
    expandedHalfShape.insert(expandedHalfShape.begin() + axis + 1, stride);
    int32_t joinAxis = expandedHalfShape.size();
    auto order = llvm::to_vector(llvm::seq<int32_t>(joinAxis));
    order.insert(order.begin() + axis + 1, joinAxis);
    lhs = tt::ReshapeOp::create(rewriter, loc, expandedHalfShape, lhs);
    rhs = tt::ReshapeOp::create(rewriter, loc, expandedHalfShape, rhs);
    Value joined = tt::JoinOp::create(rewriter, loc, lhs, rhs);
    Value transposed = tt::TransOp::create(rewriter, loc, joined, order);
    return tt::ReshapeOp::create(rewriter, loc, fullShape, transposed);
  };

  auto extractLimb = [&](Value payload, ttg::DotOperandEncodingAttr layout,
                         int64_t limb) -> Value {
    auto payloadTy = cast<RankedTensorType>(payload.getType());
    auto limbTy = payloadTy.clone(i8Ty);
    auto dotLimbTy = limbTy.cloneWithEncoding(layout);
    Value shifted = payload;
    if (limb != 0) {
      Value shift = getIntConstantLike(rewriter, loc, payloadTy, 8 * limb);
      shifted = arith::ShRUIOp::create(rewriter, loc, shifted, shift);
    }
    Value truncated = arith::TruncIOp::create(rewriter, loc, limbTy, shifted);
    return ttg::ConvertLayoutOp::create(rewriter, loc, dotLimbTy, truncated);
  };

  auto emitFragments = [&](auto &&self, Value a, Value b, Value accumulator,
                           unsigned splitIdx) -> Value {
    if (splitIdx < fragmentSplits.size()) {
      auto [axis, stride] = fragmentSplits[splitIdx];
      auto aHalves =
          axis == 0 ? splitAtRegisterBasis(a, axis, stride) : std::pair{a, a};
      auto bHalves =
          axis == 1 ? splitAtRegisterBasis(b, axis, stride) : std::pair{b, b};
      auto accHalves = splitAtRegisterBasis(accumulator, axis, stride);
      Value lhs = self(self, aHalves.first, bHalves.first, accHalves.first,
                       splitIdx + 1);
      Value rhs = self(self, aHalves.second, bHalves.second, accHalves.second,
                       splitIdx + 1);
      return joinAtRegisterBasis(lhs, rhs, axis, stride);
    }

    auto tileWorkTy = cast<RankedTensorType>(accumulator.getType())
                          .cloneWithEncoding(mmaLayout);
    auto accMmaTy = tileWorkTy.clone(i32Ty);
    accumulator =
        ttg::ConvertLayoutOp::create(rewriter, loc, tileWorkTy, accumulator);

    for (int64_t diagonal = 0; diagonal <= highestDiagonal; ++diagonal) {
      bool accumulateDirectly = diagonal == 0 && workElem == i32Ty;
      Value diagonalSum = accumulateDirectly
                              ? accumulator
                              : getIntConstantLike(rewriter, loc, accMmaTy, 0);
      int64_t firstALimb = std::max<int64_t>(0, diagonal - bLimbs + 1);
      int64_t lastALimb = std::min(diagonal, aLimbs - 1);
      for (int64_t aLimb = firstALimb; aLimb <= lastALimb; ++aLimb) {
        int64_t bLimb = diagonal - aLimb;
        Value aLimbValue = extractLimb(a, aDotLayout, aLimb);
        Value bLimbValue = extractLimb(b, bDotLayout, bLimb);
        diagonalSum = DotI8Op::create(rewriter, loc, accMmaTy, aLimbValue,
                                      bLimbValue, diagonalSum,
                                      aLimb == aLimbs - 1, bLimb == bLimbs - 1);
      }
      if (accumulateDirectly) {
        accumulator = diagonalSum;
        continue;
      }

      Value contribution = diagonalSum;
      if (accElem.getWidth() == 64) {
        // K32 keeps each completed diagonal exact in i32 before extension.
        contribution =
            arith::ExtSIOp::create(rewriter, loc, tileWorkTy, contribution);
      }
      if (diagonal != 0) {
        Value shift =
            getIntConstantLike(rewriter, loc, tileWorkTy, 8 * diagonal);
        contribution =
            arith::ShLIOp::create(rewriter, loc, contribution, shift);
      }
      accumulator =
          arith::AddIOp::create(rewriter, loc, accumulator, contribution);
    }
    return accumulator;
  };

  Value product =
      emitFragments(emitFragments, aPayload, bPayload, initialAccumulator, 0);
  return ttg::ConvertLayoutOp::create(rewriter, loc, workMmaTy, product);
}

std::optional<scf::ForOp> emitMmaEmulationLoops(
    PatternRewriter &rewriter, Location loc, const MmaOperandSource &aSource,
    const MmaOperandSource &bSource, Value dPtr, int64_t m, int64_t n,
    int64_t k, int64_t tileM, int64_t tileN, RankedTensorType accTileTy,
    ttg::DistributedEncodingTrait accLayout, IntegerType accElem, Value useDInt,
    Value predInt, int64_t dStride, const DotScaleConfig &scale = {},
    int64_t dRowStride = 1) {
  if ((m % tileM) != 0 || (n % tileN) != 0)
    return std::nullopt;

  OpBuilder::InsertionGuard guard(rewriter);
  auto i32Ty = rewriter.getI32Type();
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
  auto nLoop = scf::ForOp::create(rewriter, loc, zero, nUpper, nStep);
  rewriter.setInsertionPointToStart(nLoop.getBody());
  Value mIdxI32 =
      arith::IndexCastOp::create(rewriter, loc, i32Ty, mLoop.getInductionVar());
  Value nIdxI32 =
      arith::IndexCastOp::create(rewriter, loc, i32Ty, nLoop.getInductionVar());

  Value dRowStrideConst = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(dRowStride));
  Value dStrideConst = arith::ConstantOp::create(
      rewriter, loc, rewriter.getI32IntegerAttr(dStride));

  Value mDOffset =
      arith::MulIOp::create(rewriter, loc, mIdxI32, dRowStrideConst);
  Value nDOffset = arith::MulIOp::create(rewriter, loc, nIdxI32, dStrideConst);
  Value dOffset = arith::AddIOp::create(rewriter, loc, mDOffset, nDOffset);
  Value dTilePtr =
      tt::AddPtrOp::create(rewriter, loc, dPtr.getType(), dPtr, dOffset);

  Value sum;
  int numWarps = ttg::lookupNumWarps(rewriter.getInsertionBlock()->getParent());
  auto isScaleK32Aligned = [](Value scalePtr, int64_t scaleFactor) {
    return !scalePtr || (scaleFactor > 0 && ((kI8MmaK % scaleFactor) == 0 ||
                                             (scaleFactor % kI8MmaK) == 0));
  };
  bool canUseI8Decomposition =
      (k % kI8MmaK) == 0 && supportsI8DotDecomposition(rewriter, accElem) &&
      isScaleK32Aligned(scale.aScalePtr, scale.aScaleFactor) &&
      isScaleK32Aligned(scale.bScalePtr, scale.bScaleFactor) &&
      canUseI8MmaTile(tileM, tileN, numWarps);
  ttg::NvidiaMmaEncodingAttr mmaLayout;
  if (canUseI8Decomposition) {
    auto warpsPerCTA = ttg::getMmaV2WarpsPerCTA(accTileTy.getShape(), numWarps);
    mmaLayout = ttg::NvidiaMmaEncodingAttr::get(
        rewriter.getContext(), /*versionMajor=*/2, /*versionMinor=*/0,
        warpsPerCTA, ttg::getCGALayout(accLayout),
        SmallVector<unsigned>{kI8MmaM, kI8MmaN});
    accTileTy = accTileTy.cloneWithEncoding(mmaLayout);
  }
  auto accTileITy = getScratchStorageType(accTileTy);
  Value accTile = loadScratchStrided2D(rewriter, loc, dTilePtr, accTileTy,
                                       dRowStride, dStride);
  Value accTileI = embedToInt(rewriter, loc, accTile);
  if (canUseI8Decomposition) {
    auto workElem = accElem.getWidth() == 64 ? accElem : rewriter.getI32Type();
    accTileI = castSignedIntValueToType(rewriter, loc, accTileI,
                                        accTileITy.clone(workElem));
  }
  Value useDMask =
      castScalarIntToIntLike(rewriter, loc, useDInt, accTileI.getType());
  Value accInit = arith::MulIOp::create(rewriter, loc, accTileI, useDMask);

  if (canUseI8Decomposition) {
    Value kUpper =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(k));
    Value kStep = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI32IntegerAttr(kI8MmaK));
    auto kLoop =
        scf::ForOp::create(rewriter, loc, zero, kUpper, kStep, accInit);
    rewriter.setInsertionPointToStart(kLoop.getBody());
    Value kIdx = kLoop.getInductionVar();
    Value kI32 = arith::IndexCastOp::create(rewriter, loc, i32Ty, kIdx);
    Value aChunk;
    Value bChunk;
    if (scale.computeElem) {
      aChunk = loadScaledOperandK32(rewriter, loc, /*isLhs=*/true, aSource,
                                    scale, mIdxI32, kI32, mmaLayout);
      bChunk = loadScaledOperandK32(rewriter, loc, /*isLhs=*/false, bSource,
                                    scale, nIdxI32, kI32, mmaLayout);
    } else {
      aChunk = loadOperandK32(rewriter, loc, /*isLhs=*/true, aSource, mIdxI32,
                              kI32, mmaLayout);
      bChunk = loadOperandK32(rewriter, loc, /*isLhs=*/false, bSource, nIdxI32,
                              kI32, mmaLayout);
    }
    Value next =
        emitI8DotDecomposition(rewriter, loc, embedToInt(rewriter, loc, aChunk),
                               embedToInt(rewriter, loc, bChunk), accElem,
                               kLoop.getRegionIterArgs()[0]);
    scf::YieldOp::create(rewriter, loc, next);
    rewriter.setInsertionPointAfter(kLoop);
    sum = kLoop.getResult(0);
  } else {
    auto aSliceTy = RankedTensorType::get(
        {tileM, 1}, aSource.tileType.getElementType(), accLayout);
    auto bSliceTy = RankedTensorType::get(
        {1, tileN}, bSource.tileType.getElementType(), accLayout);
    Value zeroSum = getIntConstantLike(rewriter, loc, accTileITy, 0);
    Value kUpper =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(k));
    Value kStep =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(1));
    auto kLoop =
        scf::ForOp::create(rewriter, loc, zero, kUpper, kStep, zeroSum);
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
    Value aSlice = loadMmaOperand(rewriter, loc, aSource, aSliceTy,
                                  /*isLhs=*/true, mIdxI32, aKIdx);
    Value bSlice = loadMmaOperand(rewriter, loc, bSource, bSliceTy,
                                  /*isLhs=*/false, nIdxI32, bKIdx);
    if (scale.aKPackFactor == 2)
      aSlice = unpackPackedFp4Slice(rewriter, loc, aSlice, kI32);
    if (scale.bKPackFactor == 2)
      bSlice = unpackPackedFp4Slice(rewriter, loc, bSlice, kI32);
    Value aScaleSlice;
    if (scale.aScalePtr) {
      aScaleSlice =
          loadScaleSlice(rewriter, loc, /*isLhs=*/true, scale, mIdxI32, kI32);
    }
    Value bScaleSlice;
    if (scale.bScalePtr) {
      bScaleSlice =
          loadScaleSlice(rewriter, loc, /*isLhs=*/false, scale, nIdxI32, kI32);
    }
    Value partial =
        emulateDotStep(rewriter, loc, aSlice, bSlice, aScaleSlice, bScaleSlice,
                       tileM, tileN, accLayout, accElem, scale);
    Value acc = kLoop.getRegionIterArgs()[0];
    Value next = arith::AddIOp::create(rewriter, loc, acc, partial);
    scf::YieldOp::create(rewriter, loc, next);
    rewriter.setInsertionPointAfter(kLoop);
    sum = kLoop.getResult(0);
    sum = arith::AddIOp::create(rewriter, loc, sum, accInit);
  }

  Value predMask =
      castScalarIntToIntLike(rewriter, loc, predInt, accTileI.getType());
  Value oneI = getIntConstantLike(rewriter, loc, accTileI.getType(), 1);
  Value predInv = arith::SubIOp::create(rewriter, loc, oneI, predMask);
  Value outMasked = arith::MulIOp::create(rewriter, loc, sum, predMask);
  Value accMasked = arith::MulIOp::create(rewriter, loc, accTileI, predInv);
  Value outSelI = arith::AddIOp::create(rewriter, loc, outMasked, accMasked);
  outSelI = castSignedIntValueToType(rewriter, loc, outSelI, accTileITy);
  Value out = isFloatLike(accTileTy)
                  ? unembedToFloat(rewriter, loc, outSelI, accTileTy)
                  : outSelI;
  createGlobalScratchBarrier(rewriter, loc);
  storeScratchStrided2D(rewriter, loc, dTilePtr, out, accTileTy, dRowStride,
                        dStride);
  return mLoop;
}

std::optional<scf::ForOp> emitMmaEmulationLoops(
    PatternRewriter &rewriter, Location loc, Value aPtr, Value bPtr, Value dPtr,
    int64_t m, int64_t n, int64_t k, int64_t tileM, int64_t tileN,
    RankedTensorType aTileTy, RankedTensorType bTileTy,
    RankedTensorType accTileTy, ttg::DistributedEncodingTrait accLayout,
    IntegerType accElem, Value useDInt, Value predInt, int64_t aStride,
    int64_t bStride, int64_t dStride, const DotScaleConfig &scale = {},
    int64_t aRowStride = 1, int64_t bRowStride = 1, int64_t dRowStride = 1) {
  MmaOperandSource aSource{aPtr, Value(), aTileTy, aRowStride, aStride};
  MmaOperandSource bSource{bPtr, Value(), bTileTy, bRowStride, bStride};
  return emitMmaEmulationLoops(rewriter, loc, aSource, bSource, dPtr, m, n, k,
                               tileM, tileN, accTileTy, accLayout, accElem,
                               useDInt, predInt, dStride, scale, dRowStride);
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
    auto lhsI = embedToInt(rewriter, loc, op.getLhs());
    auto rhsI = embedToInt(rewriter, loc, op.getRhs());
    auto resI = OpI::create(rewriter, loc, lhsI, rhsI);
    auto resF = unembedToFloat(rewriter, loc, resI, op.getType());
    rewriter.replaceOp(op, resF);
    return success();
  }
};

struct NegFOpPattern : public OpRewritePattern<arith::NegFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::NegFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();

    auto loc = op.getLoc();
    auto inputI = embedToInt(rewriter, loc, op.getOperand());
    auto zeroI = getIntConstantLike(rewriter, loc, inputI.getType(), 0);
    auto resI = arith::SubIOp::create(rewriter, loc, zeroI, inputI);
    auto resF = unembedToFloat(rewriter, loc, resI, op.getType());
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
    rewriter.replaceOp(
        op, fpsanFma(rewriter, op.getLoc(), op.getA(), op.getB(), op.getC()));
    return success();
  }
};

struct ExpOpPattern : public OpRewritePattern<math::ExpOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    Value result = fpsanExp(rewriter, op.getLoc(), op.getOperand());
    if (!result)
      result = fpsanUnaryTagged(rewriter, op.getLoc(), op.getOperand(),
                                UnaryOpId::Exp);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct Exp2OpPattern : public OpRewritePattern<math::Exp2Op> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::Exp2Op op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    Value result = fpsanExp2(rewriter, op.getLoc(), op.getOperand());
    if (!result)
      result = fpsanUnaryTagged(rewriter, op.getLoc(), op.getOperand(),
                                UnaryOpId::Exp2);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct CosOpPattern : public OpRewritePattern<math::CosOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::CosOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    Value result = fpsanCos(rewriter, op.getLoc(), op.getOperand());
    if (!result)
      result = fpsanUnaryTagged(rewriter, op.getLoc(), op.getOperand(),
                                UnaryOpId::Cos);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SinOpPattern : public OpRewritePattern<math::SinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(math::SinOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFloatLike(op.getType()))
      return failure();
    Value result = fpsanSin(rewriter, op.getLoc(), op.getOperand());
    if (!result)
      result = fpsanUnaryTagged(rewriter, op.getLoc(), op.getOperand(),
                                UnaryOpId::Sin);
    rewriter.replaceOp(op, result);
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
    auto inI = embedToInt(rewriter, loc, op.getIn());
    auto outI = castSignedIntValueToType(rewriter, loc, inI,
                                         getIntTypeLike(op.getType()));
    auto outF = unembedToFloat(rewriter, loc, outI, op.getType());
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
    auto inI = embedToInt(rewriter, loc, op.getIn());
    auto outI = castSignedIntValueToType(rewriter, loc, inI,
                                         getIntTypeLike(op.getType()));
    auto outF = unembedToFloat(rewriter, loc, outI, op.getType());
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
    auto inI = embedToInt(rewriter, loc, op.getSrc());
    auto outI = castSignedIntValueToType(rewriter, loc, inI,
                                         getIntTypeLike(op.getType()));
    auto outF = unembedToFloat(rewriter, loc, outI, op.getType());
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
      return emitFpSanInvariantError(op.getOperation());
    auto srcElemTy = dyn_cast<IntegerType>(srcTy.getElementType());
    if (!srcElemTy || srcElemTy.getWidth() != 8)
      return emitFpSanInvariantError(op.getOperation());

    auto dstIntTy = cast<RankedTensorType>(getIntTypeLike(dstTy));
    auto loc = op.getLoc();
    Value result = unpackPackedFp4Tensor(rewriter, loc, op.getSrc(),
                                         op.getAxis(), dstIntTy);
    rewriter.replaceOp(op, unembedToFloat(rewriter, loc, result, dstTy));
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
      return emitFpSanInvariantError(op.getOperation());
    if (aTy.getRank() != bTy.getRank() || aTy.getRank() != cTy.getRank() ||
        (aTy.getRank() != 2 && aTy.getRank() != 3))
      return emitFpSanUnsupported(op.getOperation());
    if (!aTy.getEncoding() || !bTy.getEncoding() || !cTy.getEncoding())
      return emitFpSanUnsupported(op.getOperation());

    auto aShape = aTy.getShape();
    auto bShape = bTy.getShape();
    auto cShape = cTy.getShape();
    auto loc = op.getLoc();
    int64_t batch = 1;
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t aBatchStride = 0;
    int64_t bBatchStride = 0;
    int64_t dBatchStride = 0;
    int64_t aRowStride;
    int64_t aKStride;
    int64_t bKStride;
    int64_t bNStride;
    int64_t dRowStride;
    int64_t dNStride;
    if (aTy.getRank() == 2) {
      if (aShape[1] != bShape[0] || aShape[0] != cShape[0] ||
          bShape[1] != cShape[1])
        return emitFpSanInvariantError(op.getOperation());
      m = aShape[0];
      k = aShape[1];
      n = bShape[1];
      aRowStride = 1;
      aKStride = m;
      bKStride = 1;
      bNStride = k;
      dRowStride = 1;
      dNStride = m;
    } else {
      if (aShape[0] != bShape[0] || aShape[0] != cShape[0] ||
          aShape[2] != bShape[1] || aShape[1] != cShape[1] ||
          bShape[2] != cShape[2])
        return emitFpSanInvariantError(op.getOperation());
      batch = aShape[0];
      m = aShape[1];
      k = aShape[2];
      n = bShape[2];
      aBatchStride = 1;
      bBatchStride = 1;
      dBatchStride = 1;
      aRowStride = batch;
      aKStride = batch * m;
      bKStride = batch;
      bNStride = batch * k;
      dRowStride = batch;
      dNStride = batch * m;
    }

    auto accElem = IntegerType::get(
        rewriter.getContext(), cTy.getElementType().getIntOrFloatBitWidth());
    Value useDInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));
    Value predInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));

    auto [tileM, tileN] = getMmaEmulationTileShape(rewriter, m, n, k, accElem);

    // Use optimized blocked layouts for emulation tiles instead of the
    // original dot encodings.  Encodings like AMDWmmaEncodingAttr impose
    // minimum shape requirements that FMA fallback tiles cannot satisfy.
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
    if (!aPtr || !bPtr || !dPtr)
      return emitFpSanCodegenError(op.getOperation());

    // Each warp may only store a subset of each tile's rows, so a barrier is
    // needed to make all scratch stores visible before the loops read them.
    createGlobalScratchBarrier(rewriter, loc);

    if (batch == 1) {
      auto mLoop = emitMmaEmulationLoops(
          rewriter, loc, aPtr, bPtr, dPtr, m, n, k, tileM, tileN, aTileTy,
          bTileTy, accTileTy, accLayout, accElem, useDInt, predInt,
          /*aStride=*/aKStride, /*bStride=*/bNStride, /*dStride=*/dNStride,
          /*scale=*/{}, aRowStride, bKStride, dRowStride);
      if (!mLoop)
        return emitFpSanUnsupported(op.getOperation());
      rewriter.setInsertionPointAfter(*mLoop);
    } else {
      Value zero = arith::ConstantOp::create(rewriter, loc,
                                             rewriter.getI32IntegerAttr(0));
      Value batchUpper = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI32IntegerAttr(batch));
      Value one = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(1));
      auto batchLoop = scf::ForOp::create(rewriter, loc, zero, batchUpper, one);
      rewriter.setInsertionPointToStart(batchLoop.getBody());
      Value batchIdx = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI32Type(), batchLoop.getInductionVar());
      Value aBatchOffset = arith::MulIOp::create(
          rewriter, loc, batchIdx,
          arith::ConstantOp::create(rewriter, loc,
                                    rewriter.getI32IntegerAttr(aBatchStride)));
      Value bBatchOffset = arith::MulIOp::create(
          rewriter, loc, batchIdx,
          arith::ConstantOp::create(rewriter, loc,
                                    rewriter.getI32IntegerAttr(bBatchStride)));
      Value dBatchOffset = arith::MulIOp::create(
          rewriter, loc, batchIdx,
          arith::ConstantOp::create(rewriter, loc,
                                    rewriter.getI32IntegerAttr(dBatchStride)));
      Value aBatchPtr = tt::AddPtrOp::create(rewriter, loc, aPtr.getType(),
                                             aPtr, aBatchOffset);
      Value bBatchPtr = tt::AddPtrOp::create(rewriter, loc, bPtr.getType(),
                                             bPtr, bBatchOffset);
      Value dBatchPtr = tt::AddPtrOp::create(rewriter, loc, dPtr.getType(),
                                             dPtr, dBatchOffset);
      auto mLoop = emitMmaEmulationLoops(
          rewriter, loc, aBatchPtr, bBatchPtr, dBatchPtr, m, n, k, tileM, tileN,
          aTileTy, bTileTy, accTileTy, accLayout, accElem, useDInt, predInt,
          /*aStride=*/aKStride, /*bStride=*/bNStride,
          /*dStride=*/dNStride, /*scale=*/{}, aRowStride, bKStride, dRowStride);
      if (!mLoop)
        return emitFpSanUnsupported(op.getOperation());
      rewriter.setInsertionPointAfter(batchLoop);
    }

    // Same reason: each warp may only write a subset of D's rows in the loop,
    // so synchronize before the final load.
    createGlobalScratchBarrier(rewriter, loc);

    Value out = aTy.getRank() == 2
                    ? loadScratchStrided2D(rewriter, loc, dPtr, cTy,
                                           /*stride1=*/m)
                    : loadFpSanScratchMemory(rewriter, loc, dPtr, cTy);
    if (!out)
      return emitFpSanCodegenError(op.getOperation());
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
      return emitFpSanInvariantError(op.getOperation());
    if (aTy.getRank() != 2 || bTy.getRank() != 2 || cTy.getRank() != 2 ||
        (aScale && aScaleTy.getRank() != 2) ||
        (bScale && bScaleTy.getRank() != 2))
      return emitFpSanUnsupported(op.getOperation());
    if (!aTy.getEncoding() || !bTy.getEncoding() || !cTy.getEncoding() ||
        (aScale && !aScaleTy.getEncoding()) ||
        (bScale && !bScaleTy.getEncoding()))
      return emitFpSanUnsupported(op.getOperation());
    // TODO: Support M/N packing.
    if (!op.getLhsKPack() || !op.getRhsKPack())
      return emitFpSanUnsupported(op.getOperation());

    auto aShape = aTy.getShape();
    auto bShape = bTy.getShape();
    auto cShape = cTy.getShape();
    if (aShape[0] != cShape[0] || bShape[1] != cShape[1])
      return emitFpSanInvariantError(op.getOperation());

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
      return emitFpSanInvariantError(op.getOperation());

    auto loc = op.getLoc();
    int64_t m = cShape[0];
    int64_t n = cShape[1];
    if ((aScale && aScaleTy.getShape()[0] != m) ||
        (bScale && bScaleTy.getShape()[0] != n))
      return emitFpSanInvariantError(op.getOperation());

    auto accElem = IntegerType::get(
        rewriter.getContext(), cTy.getElementType().getIntOrFloatBitWidth());
    Value useDInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));
    Value predInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));

    auto [tileM, tileN] = getMmaEmulationTileShape(rewriter, m, n, k, accElem);

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
    if (!aPtr || !bPtr || !dPtr)
      return emitFpSanCodegenError(op.getOperation());

    auto aElemType = op.getAElemType();
    auto bElemType = op.getBElemType();
    bool skipAScale = aElemType == tt::ScaleDotElemType::BF16 ||
                      aElemType == tt::ScaleDotElemType::FP16;
    bool skipBScale = bElemType == tt::ScaleDotElemType::BF16 ||
                      bElemType == tt::ScaleDotElemType::FP16;

    DotScaleConfig scale;
    scale.aElemType = aElemType;
    scale.bElemType = bElemType;
    scale.computeElem =
        getDotScaledComputeFloatType(rewriter, aElemType, bElemType);
    scale.aKPackFactor = aKPackFactor;
    scale.bKPackFactor = bKPackFactor;
    if (aScale && !skipAScale) {
      scale.aScalePtr = createScratchAndStore(rewriter, loc, aScale, aScaleTy);
      if (!scale.aScalePtr)
        return emitFpSanCodegenError(op.getOperation());
      scale.aScaleStride = aScaleTy.getShape()[0];
      scale.aScaleFactor = op.deduceScaleFactor();
      scale.aScaleTileTy = RankedTensorType::get(
          {tileM, 1}, aScaleTy.getElementType(), accLayout);
    }
    if (bScale && !skipBScale) {
      scale.bScalePtr = createScratchAndStore(rewriter, loc, bScale, bScaleTy);
      if (!scale.bScalePtr)
        return emitFpSanCodegenError(op.getOperation());
      scale.bScaleStride = bScaleTy.getShape()[0];
      scale.bScaleFactor = op.deduceScaleFactor();
      scale.bScaleTileTy = RankedTensorType::get(
          {1, tileN}, bScaleTy.getElementType(), accLayout);
    }

    createGlobalScratchBarrier(rewriter, loc);

    auto mLoop = emitMmaEmulationLoops(
        rewriter, loc, aPtr, bPtr, dPtr, m, n, k, tileM, tileN, aTileTy,
        bTileTy, accTileTy, accLayout, accElem, useDInt, predInt,
        /*aStride=*/m, /*bStride=*/bPackedK, /*dStride=*/m, scale);
    if (!mLoop)
      return emitFpSanUnsupported(op.getOperation());
    rewriter.setInsertionPointAfter(*mLoop);

    createGlobalScratchBarrier(rewriter, loc);

    Value out = loadScratchStrided2D(rewriter, loc, dPtr, cTy, /*stride1=*/m);
    if (!out)
      return emitFpSanCodegenError(op.getOperation());
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
      return emitFpSanCodegenError(op.getOperation());

    Location loc = op.getLoc();
    auto resultTy = cast<RankedTensorType>(op.getResult().getType());
    if (!resultTy.getEncoding())
      return emitFpSanUnsupported(op.getOperation());

    Value result;
    if (info->scaleSourceType) {
      auto scaleMemTy = info->scaleSourceType;
      auto physicalMemTy = cast<ttg::MemDescType>(op.getSrc().getType());
      auto scaleShape = scaleMemTy.getShape();

      constexpr int64_t physicalRows = 128;
      auto physicalEncoding =
          cast<ttng::TensorMemoryEncodingAttr>(physicalMemTy.getEncoding());
      int64_t rows = scaleShape[0];
      int64_t cols = scaleShape[1];
      SmallVector<int64_t> physicalShape = {physicalRows, rows * cols / 32};
      if (rows % 32 != 0 || cols % 4 != 0 ||
          ttng::getTmemAllocSizes(scaleMemTy).numRows != physicalRows ||
          physicalMemTy.getElementType().getIntOrFloatBitWidth() != 8 ||
          physicalMemTy.getShape() != ArrayRef<int64_t>(physicalShape) ||
          physicalEncoding.getBlockM() != physicalRows ||
          physicalEncoding.getColStride() != 1)
        return emitFpSanUnsupported(op.getOperation());

      // Reconstruct the scale-copy hardware representation from the latest
      // compact shadow at the point where the raw TMEM alias is consumed.
      result = createLoadScratchMemory(rewriter, loc, info->ptr,
                                       getScratchStorageType(info->tensorType));
      SmallVector<int64_t> shape = {rows / 32, 32, cols / 4, 4};
      result = tt::ReshapeOp::create(rewriter, loc, shape, result);
      result = tt::TransOp::create(rewriter, loc, result, {1, 2, 0, 3});
      shape = {32, physicalShape[1]};
      result = tt::ReshapeOp::create(rewriter, loc, shape, result);
      shape = {1, 32, physicalShape[1]};
      result = tt::ReshapeOp::create(rewriter, loc, shape, result);
      shape[0] = 4;
      auto repeatedTy = cast<RankedTensorType>(result.getType()).clone(shape);
      result = tt::BroadcastOp::create(rewriter, loc, repeatedTy, result);
      result = tt::ReshapeOp::create(rewriter, loc, physicalShape, result);
      result = ttg::ConvertLayoutOp::create(
          rewriter, loc, getScratchStorageType(resultTy), result);
      if (isFloatLike(resultTy))
        result = unembedToFloat(rewriter, loc, result, resultTy);
    } else {
      result = loadFpSanScratchMemory(rewriter, loc, info->ptr, resultTy);
    }

    if (!result)
      return emitFpSanCodegenError(op.getOperation());

    Value reduced;
    if (op.getRed()) {
      Value reductionInput = result;
      if (op.getAbs().value_or(false))
        reductionInput =
            math::AbsFOp::create(rewriter, loc, reductionInput).getResult();
      reductionInput = embedToInt(rewriter, loc, reductionInput);

      auto reduce =
          tt::ReduceOp::create(rewriter, loc, ValueRange{reductionInput}, 1);
      Block &block = reduce.getCombineOp().emplaceBlock();
      Type elemTy = getElementTypeOrSelf(reductionInput.getType());
      block.addArguments({elemTy, elemTy}, {loc, loc});

      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(&block);
        Value lhs = block.getArgument(0);
        Value rhs = block.getArgument(1);
        // FPSAN gives both NaN modes the same signed-payload semantics.
        Value combined =
            *op.getRedOp() == ttng::TMEMLoadReduceModifier::MIN
                ? arith::MinSIOp::create(rewriter, loc, lhs, rhs).getResult()
                : arith::MaxSIOp::create(rewriter, loc, lhs, rhs).getResult();
        tt::ReduceReturnOp::create(rewriter, loc, ValueRange{combined});
      }
      reduced = unembedToFloat(rewriter, loc, reduce.getResult().front(),
                               op.getRed().getType());
    }

    createGlobalScratchBarrier(rewriter, loc,
                               scratch->usesSharedClusterState());

    SmallVector<Value> replacements{result};
    if (op.getToken()) {
      SmallVector<Value> deps;
      if (op.getDep())
        deps.push_back(op.getDep());
      replacements.push_back(createAsyncToken(rewriter, loc, deps));
    }
    if (reduced)
      replacements.push_back(reduced);
    rewriter.replaceOp(op, replacements);
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
      return emitFpSanCodegenError(op.getOperation());
    if (info->scaleSourceType)
      return emitFpSanUnsupported(op.getOperation());

    auto loc = op.getLoc();
    auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTy.getEncoding())
      return emitFpSanUnsupported(op.getOperation());
    auto storageTy = getScratchStorageType(srcTy);
    Value stored = embedToInt(rewriter, loc, op.getSrc());
    if (!matchPattern(op.getPred(), m_One())) {
      Value previous =
          createLoadScratchMemory(rewriter, loc, info->ptr, storageTy);
      Value pred = castScalarIntToIntLike(
          rewriter, loc, op.getPred(), storageTy.clone(rewriter.getI1Type()));
      stored = arith::SelectOp::create(rewriter, loc, pred, stored, previous);
    }
    if (!createStoreScratchMemory(rewriter, loc, info->ptr, stored, storageTy))
      return emitFpSanCodegenError(op.getOperation());

    createGlobalScratchBarrier(rewriter, loc,
                               scratch->usesSharedClusterState());

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
      return emitFpSanCodegenError(op.getOperation());
    if (info->scaleSourceType)
      return emitFpSanUnsupported(op.getOperation());

    auto loc = op.getLoc();
    auto srcMemTy = cast<ttg::MemDescType>(op.getSrc().getType());
    auto dstMemTy = cast<ttg::MemDescType>(op.getDst().getType());
    auto srcEncoding =
        scratch->getScratchEncoding(rewriter, op.getDst(), dstMemTy);
    auto srcRegTy = RankedTensorType::get(
        srcMemTy.getShape(), srcMemTy.getElementType(), srcEncoding);
    if (scratch->usesSharedClusterState()) {
      // Match the lead-CTA TMA wait before both CTAs emulate the TMEM copy.
      ttng::ClusterBarrierOp::create(rewriter, loc);
    }
    Value srcReg =
        ttg::LocalLoadOp::create(rewriter, loc, srcRegTy, op.getSrc(), Value())
            .getResult();
    if (!storeFpSanScratchMemory(rewriter, loc, info->ptr, srcReg, srcRegTy))
      return emitFpSanCodegenError(op.getOperation());

    createGlobalScratchBarrier(rewriter, loc,
                               scratch->usesSharedClusterState());

    rewriter.eraseOp(op);
    return success();
  }

private:
  TmemScratchManager *scratch;
};

struct TCGen5CommitPattern : public OpRewritePattern<ttng::TCGen5CommitOp> {
  TCGen5CommitPattern(MLIRContext *ctx, bool twoCTAs)
      : OpRewritePattern(ctx), twoCTAs(twoCTAs) {}

  LogicalResult matchAndRewrite(ttng::TCGen5CommitOp op,
                                PatternRewriter &rewriter) const override {
    if (twoCTAs)
      createGlobalScratchBarrier(rewriter, op.getLoc(),
                                 /*sharedClusterState=*/true);
    createSynchronousCompletionArrive(rewriter, op.getLoc(), op.getBarrier(),
                                      op.getPred());
    rewriter.eraseOp(op);
    return success();
  }

private:
  bool twoCTAs;
};

struct WarpGroupDotPattern : public OpRewritePattern<ttng::WarpGroupDotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::WarpGroupDotOp op,
                                PatternRewriter &rewriter) const override {
    auto aTy = dyn_cast<ttg::TensorOrMemDesc>(op.getA().getType());
    auto bMemTy = dyn_cast<ttg::MemDescType>(op.getB().getType());
    auto cTy = dyn_cast<RankedTensorType>(op.getC().getType());
    if (!aTy || !bMemTy || !cTy)
      return emitFpSanInvariantError(op.getOperation());

    if (auto aMemTy = dyn_cast<ttg::MemDescType>(op.getA().getType())) {
      if (!isa<ttg::SharedMemorySpaceAttr>(aMemTy.getMemorySpace()))
        return emitFpSanInvariantError(op.getOperation());
    }
    if (!isa<ttg::SharedMemorySpaceAttr>(bMemTy.getMemorySpace()))
      return emitFpSanInvariantError(op.getOperation());

    bool aIsFloat = isa<FloatType>(aTy.getElementType());
    bool bIsFloat = isa<FloatType>(bMemTy.getElementType());
    bool cIsFloat = isa<FloatType>(cTy.getElementType());
    if (!aIsFloat && !bIsFloat && !cIsFloat)
      return failure();
    if (!aIsFloat || !bIsFloat || !cIsFloat)
      return emitFpSanUnsupported(op.getOperation());

    if (aTy.getRank() != 2 || bMemTy.getRank() != 2 || cTy.getRank() != 2)
      return emitFpSanUnsupported(op.getOperation());
    auto aShape = aTy.getShape();
    auto bShape = bMemTy.getShape();
    auto cShape = cTy.getShape();
    if (aShape[1] != bShape[0] || aShape[0] != cShape[0] ||
        bShape[1] != cShape[1])
      return emitFpSanInvariantError(op.getOperation());

    auto loc = op.getLoc();
    int64_t m = aShape[0];
    int64_t k = aShape[1];
    int64_t n = bShape[1];

    auto *ctx = rewriter.getContext();
    auto accElem =
        IntegerType::get(ctx, cTy.getElementType().getIntOrFloatBitWidth());
    Value useCInt;
    if (op.getUseC()) {
      useCInt = arith::ExtUIOp::create(rewriter, loc, accElem, op.getUseC());
    } else {
      useCInt = arith::ConstantOp::create(rewriter, loc,
                                          rewriter.getIntegerAttr(accElem, 1));
    }
    Value predInt = arith::ConstantOp::create(
        rewriter, loc, rewriter.getIntegerAttr(accElem, 1));

    rewriter.setInsertionPoint(op);
    auto aScratch = createWGMMAScratch(rewriter, loc, op.getA());
    auto bScratch = createWGMMAScratch(rewriter, loc, op.getB());
    Value dPtr = createScratchAndStore(rewriter, loc, op.getC(), cTy);
    if (!aScratch || !bScratch || !dPtr)
      return emitFpSanCodegenError(op.getOperation());

    auto [tileM, tileN] = getMmaEmulationTileShape(rewriter, m, n, k, accElem);

    auto accTileLayout = getOptimizedBlockedEncoding(rewriter, {tileM, tileN},
                                                     cTy.getElementType());
    auto accTileTy = RankedTensorType::get({tileM, tileN}, cTy.getElementType(),
                                           accTileLayout);
    auto aTileLayout =
        getOptimizedBlockedEncoding(rewriter, {tileM, k}, aTy.getElementType());
    auto aTileTy =
        RankedTensorType::get({tileM, k}, aTy.getElementType(), aTileLayout);
    auto bTileLayout = getOptimizedBlockedEncoding(rewriter, {k, tileN},
                                                   bMemTy.getElementType());
    auto bTileTy =
        RankedTensorType::get({k, tileN}, bMemTy.getElementType(), bTileLayout);

    createGlobalScratchBarrier(rewriter, loc);

    auto mLoop = emitMmaEmulationLoops(
        rewriter, loc, aScratch->ptr, bScratch->ptr, dPtr, m, n, k, tileM,
        tileN, aTileTy, bTileTy, accTileTy, accTileLayout, accElem, useCInt,
        predInt, /*aStride=*/m, /*bStride=*/k, /*dStride=*/m);
    if (!mLoop)
      return emitFpSanUnsupported(op.getOperation());
    rewriter.setInsertionPointAfter(*mLoop);

    createGlobalScratchBarrier(rewriter, loc);

    Value out = loadScratchStrided2D(rewriter, loc, dPtr, cTy, /*stride1=*/m);
    if (!out)
      return emitFpSanCodegenError(op.getOperation());
    rewriter.replaceOp(op, out);
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

    bool aIsFloat = isa<FloatType>(aMemTy.getElementType());
    bool bIsFloat = isa<FloatType>(bMemTy.getElementType());
    bool dIsFloat = isa<FloatType>(dMemTy.getElementType());
    if (!aIsFloat && !bIsFloat && !dIsFloat)
      return failure();
    if (!aIsFloat || !bIsFloat || !dIsFloat)
      return emitFpSanUnsupported(op.getOperation());

    auto scope = getScratchScopeRegion(op);
    auto dInfo = scratch->getOrCreate(op.getD(), rewriter, scope);
    if (!dInfo || dInfo->scaleSourceType)
      return emitFpSanCodegenError(op.getOperation());

    auto loc = op.getLoc();

    bool aIsTmem = isa<ttng::TensorMemorySpaceAttr>(aMemTy.getMemorySpace());
    bool bIsTmem = isa<ttng::TensorMemorySpaceAttr>(bMemTy.getMemorySpace());

    if ((aIsTmem && aMemTy.getRank() != 2) ||
        (bIsTmem && bMemTy.getRank() != 2) || dMemTy.getRank() != 2)
      return emitFpSanUnsupported(op.getOperation());

    auto aShape = aMemTy.getShape();
    auto bShape = bMemTy.getShape();
    if (aShape.size() != 2 || bShape.size() != 2)
      return emitFpSanInvariantError(op.getOperation());
    if (aShape[1] != bShape[0])
      return emitFpSanInvariantError(op.getOperation());
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
    auto [tileM, tileN] = getMmaEmulationTileShape(rewriter, m, n, k, accElem);
    auto accTileLayout =
        getOptimizedBlockedEncoding(rewriter, {tileM, tileN}, accElem);
    auto accTileTy =
        RankedTensorType::get({tileM, tileN}, accElem, accTileLayout);
    Type aTileElem = aIsTmem
                         ? getScratchStorageElementType(aMemTy.getElementType())
                         : aMemTy.getElementType();
    auto aTileLayout =
        getOptimizedBlockedEncoding(rewriter, {tileM, k}, aTileElem);
    auto aTileTy = RankedTensorType::get({tileM, k}, aTileElem, aTileLayout);
    Type bTileElem = bIsTmem
                         ? getScratchStorageElementType(bMemTy.getElementType())
                         : bMemTy.getElementType();
    auto bTileLayout =
        getOptimizedBlockedEncoding(rewriter, {k, tileN}, bTileElem);
    auto bTileTy = RankedTensorType::get({k, tileN}, bTileElem, bTileLayout);

    auto aSource = createMmaOperandSource(rewriter, loc, *scratch, op.getA(),
                                          aMemTy, aIsTmem, aTileTy, scope,
                                          /*rowStride=*/1, /*stride=*/m);
    auto bSource = createMmaOperandSource(rewriter, loc, *scratch, op.getB(),
                                          bMemTy, bIsTmem, bTileTy, scope,
                                          /*rowStride=*/1, /*stride=*/k);
    if (!aSource || !bSource)
      return emitFpSanCodegenError(op.getOperation());

    // TMEM and D scratch are written cooperatively. In two-CTA mode, the
    // cluster barrier also makes both CTAs' shared operands visible before the
    // first direct load.
    createGlobalScratchBarrier(rewriter, loc,
                               scratch->usesSharedClusterState());

    auto mLoop = emitMmaEmulationLoops(
        rewriter, loc, *aSource, *bSource, dInfo->ptr, m, n, k, tileM, tileN,
        accTileTy, accTileLayout, accElem, useDInt, predInt, /*dStride=*/m);
    if (!mLoop)
      return emitFpSanUnsupported(op.getOperation());
    rewriter.setInsertionPointAfter(*mLoop);

    // The emulation loop also writes D through scratch memory from multiple
    // warps, so make those stores visible before signaling completion.
    auto postLoopBarrier = createGlobalScratchBarrier(
        rewriter, loc, scratch->usesSharedClusterState());

    if (!op.getBarriers().empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(postLoopBarrier);
      auto barriers = op.getBarriers();
      auto barrierPreds = op.getBarrierPreds();
      for (size_t i = 0; i < barriers.size(); ++i) {
        Value pred =
            arith::AndIOp::create(rewriter, loc, op.getPred(), barrierPreds[i]);
        createSynchronousCompletionArrive(rewriter, loc, barriers[i], pred);
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
    auto aScaleMemTy = cast<ttg::MemDescType>(op.getAScale().getType());
    auto bScaleMemTy = cast<ttg::MemDescType>(op.getBScale().getType());

    bool aIsTmem = isa<ttng::TensorMemorySpaceAttr>(aMemTy.getMemorySpace());
    bool bIsTmem = isa<ttng::TensorMemorySpaceAttr>(bMemTy.getMemorySpace());

    if ((aIsTmem && aMemTy.getRank() != 2) ||
        (bIsTmem && bMemTy.getRank() != 2) || (aScaleMemTy.getRank() != 2) ||
        (bScaleMemTy.getRank() != 2) || dMemTy.getRank() != 2) {
      return emitFpSanUnsupported(op.getOperation());
    }

    auto aShape = aMemTy.getShape();
    auto bShape = bMemTy.getShape();
    auto dShape = dMemTy.getShape();
    auto aScaleShape = aScaleMemTy.getShape();
    auto bScaleShape = bScaleMemTy.getShape();
    if (aShape.size() != 2 || bShape.size() != 2 || dShape.size() != 2 ||
        aScaleShape.size() != 2 || bScaleShape.size() != 2)
      return emitFpSanInvariantError(op.getOperation());

    int64_t m = dShape[0];
    int64_t n = dShape[1];
    int64_t aPackedK = aShape[1];
    int64_t bPackedK = bShape[0];
    int64_t aKPackFactor = 1;
    int64_t bKPackFactor = 1;
    if (op.getAType() == tt::ScaleDotElemType::E2M1) {
      if (op.getBlockK() == aPackedK * 2) {
        aKPackFactor = 2;
      } else {
        return emitFpSanInvariantError(op.getOperation());
      }
    }
    if (op.getBType() == tt::ScaleDotElemType::E2M1) {
      if (op.getBlockK() == bPackedK * 2) {
        bKPackFactor = 2;
      } else {
        return emitFpSanInvariantError(op.getOperation());
      }
    }

    int64_t k = aPackedK * aKPackFactor;
    if (aShape[0] != m || bShape[1] != n || k != bPackedK * bKPackFactor)
      return emitFpSanInvariantError(op.getOperation());

    auto deduceScaleFactor = [&](ArrayRef<int64_t> scaleShape,
                                 int64_t rows) -> std::optional<int64_t> {
      if (scaleShape[0] != rows || scaleShape[1] <= 0 ||
          (k % scaleShape[1]) != 0)
        return std::nullopt;
      return k / scaleShape[1];
    };
    auto aScaleFactor = deduceScaleFactor(aScaleShape, m);
    auto bScaleFactor = deduceScaleFactor(bScaleShape, n);
    if (!aScaleFactor || !bScaleFactor)
      return emitFpSanInvariantError(op.getOperation());

    auto scope = getScratchScopeRegion(op);
    auto dInfo = scratch->getOrCreate(op.getD(), rewriter, scope);
    if (!dInfo || dInfo->scaleSourceType)
      return emitFpSanCodegenError(op.getOperation());

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    auto accElem =
        IntegerType::get(ctx, dMemTy.getElementType().getIntOrFloatBitWidth());
    Value useDInt =
        arith::ExtUIOp::create(rewriter, loc, accElem, op.getUseD());
    Value predInt =
        arith::ExtUIOp::create(rewriter, loc, accElem, op.getPred());

    rewriter.setInsertionPoint(op);
    auto aScaleScratch = scratch->getOrCreate(op.getAScale(), rewriter, scope);
    if (!aScaleScratch)
      return emitFpSanCodegenError(op.getOperation());
    auto bScaleScratch = scratch->getOrCreate(op.getBScale(), rewriter, scope);
    if (!bScaleScratch)
      return emitFpSanCodegenError(op.getOperation());

    auto [tileM, tileN] = getMmaEmulationTileShape(rewriter, m, n, k, accElem);

    auto accTileLayout = getOptimizedBlockedEncoding(rewriter, {tileM, tileN},
                                                     dMemTy.getElementType());
    auto accTileTy = RankedTensorType::get(
        {tileM, tileN}, dMemTy.getElementType(), accTileLayout);
    auto aTileLayout = getOptimizedBlockedEncoding(rewriter, {tileM, aPackedK},
                                                   aMemTy.getElementType());
    auto aTileTy = RankedTensorType::get({tileM, aPackedK},
                                         aMemTy.getElementType(), aTileLayout);
    auto bTileLayout = getOptimizedBlockedEncoding(rewriter, {bPackedK, tileN},
                                                   bMemTy.getElementType());
    auto bTileTy = RankedTensorType::get({bPackedK, tileN},
                                         bMemTy.getElementType(), bTileLayout);

    auto aSource = createMmaOperandSource(rewriter, loc, *scratch, op.getA(),
                                          aMemTy, aIsTmem, aTileTy, scope,
                                          /*rowStride=*/1, /*stride=*/m);
    auto bSource = createMmaOperandSource(rewriter, loc, *scratch, op.getB(),
                                          bMemTy, bIsTmem, bTileTy, scope,
                                          /*rowStride=*/1, /*stride=*/bPackedK);
    if (!aSource || !bSource)
      return emitFpSanCodegenError(op.getOperation());

    DotScaleConfig scale;
    scale.aElemType = op.getAType();
    scale.bElemType = op.getBType();
    scale.computeElem = getDotScaledComputeFloatType(rewriter, scale.aElemType,
                                                     scale.bElemType);
    scale.aScalePtr = aScaleScratch->ptr;
    scale.bScalePtr = bScaleScratch->ptr;
    scale.aScaleTileTy = RankedTensorType::get(
        {tileM, 1}, aScaleMemTy.getElementType(), accTileLayout);
    scale.bScaleTileTy = RankedTensorType::get(
        {1, tileN}, bScaleMemTy.getElementType(), accTileLayout);
    scale.aScaleStride = aScaleShape[0];
    scale.bScaleStride = bScaleShape[0];
    scale.aKPackFactor = aKPackFactor;
    scale.bKPackFactor = bKPackFactor;
    scale.aScaleFactor = *aScaleFactor;
    scale.bScaleFactor = *bScaleFactor;

    // TMEM scales and D scratch are written cooperatively. In two-CTA mode,
    // rendezvous before either CTA directly reads the shared operands.
    createGlobalScratchBarrier(rewriter, loc,
                               scratch->usesSharedClusterState());

    auto mLoop =
        emitMmaEmulationLoops(rewriter, loc, *aSource, *bSource, dInfo->ptr, m,
                              n, k, tileM, tileN, accTileTy, accTileLayout,
                              accElem, useDInt, predInt, /*dStride=*/m, scale);
    if (!mLoop)
      return emitFpSanUnsupported(op.getOperation());
    rewriter.setInsertionPointAfter(*mLoop);

    // The emulated MMA updates the accumulator scratch cooperatively as well.
    // Flush those stores before completion barriers or later TMEM loads.
    auto postLoopBarrier = createGlobalScratchBarrier(
        rewriter, loc, scratch->usesSharedClusterState());

    if (!op.getBarriers().empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(postLoopBarrier);
      auto barriers = op.getBarriers();
      auto barrierPreds = op.getBarrierPreds();
      for (size_t i = 0; i < barriers.size(); ++i) {
        Value pred =
            arith::AndIOp::create(rewriter, loc, op.getPred(), barrierPreds[i]);
        createSynchronousCompletionArrive(rewriter, loc, barriers[i], pred);
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

bool hasFp32Signature(tt::ExternElementwiseOp op, unsigned arity) {
  if (op.getNumOperands() != arity ||
      !getElementTypeOrSelf(op.getType()).isF32())
    return false;
  return llvm::all_of(op.getOperandTypes(), [](Type type) {
    return getElementTypeOrSelf(type).isF32();
  });
}

using KnownExternTransform =
    std::function<Value(PatternRewriter &, tt::ExternElementwiseOp)>;
using UnaryExternTransform =
    std::function<Value(PatternRewriter &, Location, Value)>;
using BinaryExternTransform =
    std::function<Value(PatternRewriter &, Location, Value, Value)>;
using TernaryExternTransform =
    std::function<Value(PatternRewriter &, Location, Value, Value, Value)>;

KnownExternTransform makeUnaryExternTransform(UnaryExternTransform transform) {
  return [transform](PatternRewriter &rewriter, tt::ExternElementwiseOp op) {
    if (!hasFp32Signature(op, 1))
      return Value();
    return transform(rewriter, op.getLoc(), op.getOperand(0));
  };
}

KnownExternTransform makeTaggedUnaryExternTransform(UnaryOpId opId) {
  return [opId](PatternRewriter &rewriter, tt::ExternElementwiseOp op) {
    if (!hasFp32Signature(op, 1))
      return Value();
    return fpsanUnaryTagged(rewriter, op.getLoc(), op.getOperand(0), opId);
  };
}

KnownExternTransform
makeBinaryExternTransform(BinaryExternTransform transform) {
  return [transform](PatternRewriter &rewriter, tt::ExternElementwiseOp op) {
    if (!hasFp32Signature(op, 2))
      return Value();
    return transform(rewriter, op.getLoc(), op.getOperand(0), op.getOperand(1));
  };
}

KnownExternTransform
makeTernaryExternTransform(TernaryExternTransform transform) {
  return [transform](PatternRewriter &rewriter, tt::ExternElementwiseOp op) {
    if (!hasFp32Signature(op, 3))
      return Value();
    return transform(rewriter, op.getLoc(), op.getOperand(0), op.getOperand(1),
                     op.getOperand(2));
  };
}

std::optional<KnownExternTransform> getKnownExtern(StringRef symbol) {
  return llvm::StringSwitch<std::optional<KnownExternTransform>>(symbol)
      .Case("__nv_fast_expf", makeUnaryExternTransform(fpsanExp))
      .Case("__nv_exp2f", makeUnaryExternTransform(fpsanExp2))
      .Case("__nv_logf", makeTaggedUnaryExternTransform(UnaryOpId::Log))
      .Case("__nv_log2f", makeTaggedUnaryExternTransform(UnaryOpId::Log2))
      .Case("__nv_cosf", makeUnaryExternTransform(fpsanCos))
      .Case("__nv_sinf", makeUnaryExternTransform(fpsanSin))
      .Case("__nv_rsqrtf", makeTaggedUnaryExternTransform(UnaryOpId::Rsqrt))
      .Case("__nv_erff", makeTaggedUnaryExternTransform(UnaryOpId::Erf))
      .Case("__nv_floorf", makeTaggedUnaryExternTransform(UnaryOpId::Floor))
      .Case("__nv_ceilf", makeTaggedUnaryExternTransform(UnaryOpId::Ceil))
      .Case("__nv_fsqrt_rn",
            makeTaggedUnaryExternTransform(UnaryOpId::PreciseSqrt))
      .Case("__nv_fdiv_rn", makeBinaryExternTransform(fpsanFDiv))
      .Case("__nv_fmaf", makeTernaryExternTransform(fpsanFma))
      .Default(std::nullopt);
}

struct ExternElementwisePattern
    : public OpRewritePattern<tt::ExternElementwiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tt::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getPure() || !isFloatLike(op.getType()) ||
        op.getNumOperands() == 0 || !externHasNumericOperands(op))
      return failure();

    Value result;
    // FPSan models the intended operation, so aliases may intentionally ignore
    // backend details such as flushing subnormal inputs or outputs to zero.
    if (auto transform = getKnownExtern(op.getSymbol()))
      result = (*transform)(rewriter, op);
    if (!result) {
      uint64_t hash = stableStringHash(op.getSymbol());
      result = fpsanVariadicExternTagged(rewriter, op.getLoc(), op, hash);
    }
    if (!result)
      return emitFpSanCodegenError(op.getOperation());
    rewriter.replaceOp(op, result);
    return success();
  }
};

class FpSanitizerPass
    : public impl::TritonInstrumentFpSanitizerBase<FpSanitizerPass> {
public:
  void runOnOperation() override {
    bool fpSanErrorEmitted = false;
    ScopedDiagnosticHandler diagnosticHandler(
        &getContext(), [&](Diagnostic &diagnostic) {
          if (diagnostic.getSeverity() == DiagnosticSeverity::Error)
            fpSanErrorEmitted = true;
          return failure();
        });

    bool twoCTAs = false;
    getOperation().walk(
        [&](ttng::MMAv5OpInterface op) { twoCTAs |= op.getTwoCtas(); });

    getOperation()->setAttr(ttng::AttrTwoCTAsName,
                            BoolAttr::get(&getContext(), twoCTAs));

    TmemScratchManager scratch(twoCTAs);
    RewritePatternSet patterns(&getContext());
    patterns.add<BinaryFloatToIntPattern<arith::AddFOp, arith::AddIOp>,
                 BinaryFloatToIntPattern<arith::SubFOp, arith::SubIOp>,
                 BinaryFloatToIntPattern<arith::MulFOp, arith::MulIOp>,
                 BinaryFloatToIntPattern<arith::MinimumFOp, arith::MinSIOp>,
                 BinaryFloatToIntPattern<arith::MaximumFOp, arith::MaxSIOp>,
                 BinaryFloatToIntPattern<arith::MinNumFOp, arith::MinSIOp>,
                 BinaryFloatToIntPattern<arith::MaxNumFOp, arith::MaxSIOp>,
                 NegFOpPattern, DivFOpPattern, PreciseDivFOpPattern,
                 RemFOpPattern, FmaPattern, ExpOpPattern, Exp2OpPattern,
                 CosOpPattern, SinOpPattern, ExtFOpPattern, TruncFOpPattern,
                 FpToFpPattern, Fp4ToFpPattern, DotPattern, DotScaledPattern>(
        &getContext());
    patterns.add<UnaryPattern<math::LogOp>>(&getContext(), UnaryOpId::Log);
    patterns.add<UnaryPattern<math::Log2Op>>(&getContext(), UnaryOpId::Log2);
    patterns.add<UnaryPattern<math::SqrtOp>>(&getContext(), UnaryOpId::Sqrt);
    patterns.add<UnaryPattern<math::RsqrtOp>>(&getContext(), UnaryOpId::Rsqrt);
    patterns.add<UnaryPattern<math::ErfOp>>(&getContext(), UnaryOpId::Erf);
    patterns.add<UnaryPattern<math::FloorOp>>(&getContext(), UnaryOpId::Floor);
    patterns.add<UnaryPattern<math::CeilOp>>(&getContext(), UnaryOpId::Ceil);
    patterns.add<UnaryPattern<tt::PreciseSqrtOp>>(&getContext(),
                                                  UnaryOpId::PreciseSqrt);
    patterns.add<ExternElementwisePattern>(&getContext());
    patterns.add<TMEMLoadPattern, TMEMStorePattern, TMEMCopyPattern,
                 TCGen5MMAPattern, TCGen5MMAScaledPattern>(&getContext(),
                                                           &scratch);
    patterns.add<WarpGroupDotPattern>(&getContext());
    patterns.add<TCGen5CommitPattern>(&getContext(), twoCTAs);

    LogicalResult result =
        applyPatternsGreedily(getOperation(), std::move(patterns));
    if (failed(result)) {
      llvm::errs() << "FpSanitizer error: Failed to apply patterns\n";
      signalPassFailure();
    }
    if (fpSanErrorEmitted)
      signalPassFailure();

    // TODO: Remove unused tmem usages. This requires unwiring them from the
    // warp specialize partitions.
  }
};

} // namespace

} // namespace instrument
} // namespace triton
} // namespace mlir
