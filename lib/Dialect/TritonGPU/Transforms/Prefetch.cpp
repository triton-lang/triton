//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot.
// Those ConvertLayoutOps will be lowered to shared memory loads.
//
// For example:
// %a: tensor<128x32xf16, #enc>
// scf.for %iv = ... iter_args(%a_arg = %a, ...) {
//   %d = tt.dot %a_arg, %b, %c
//   ...
//   scf.yield %a_next, ...
// }
//
// will be translated to
//
// %a: tensor<128x32xf16, #enc>
// %a_tmp = tensor.subview %a[0, 0] [128, 16]
// %a_prefetch = ttg.local_load %a_tmp
// scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
// {
//   %x = tt.dot %a_prefetch_arg, %b, %c
//   %a_tmp_rem = tensor.subview %a_buf[0, 16] [128, 16]
//   %a_prefetch_next = ttg.local_load %a_tmp_rem
//   ...
//   scf.yield %next_a, ..., %a_prefetch_next
// }
//
// To control the degree of prefetching, the dots are sliced along M, N, K
// according to prefetchWidthM,N,K. Because the dot is 3D sliced,
// many sliced dots and sliced local_loads are created,
// and the they are emitted such that there is 1 sliced dot between
// local_loads and the dot which depends on them; data for the 0th dot is moved
// to the end of the previous iteration.
//
// Currently, only a single dot per loop is supported.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"
#include <memory>
#include <tuple>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritongpu-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

// Prefetching *all* the A, B operands of dots from local memory would be
// prohibitively expensive in terms of registers and cycles.
// This supports slicing along K (which doesn't change the D, C operands
// of the dot) and slicing along M and N.

// Store the encodings between splits and joins (along M, N) of dots to ensure
// re-joining does the inverse of splitting.
SmallVector<RankedTensorType> typesBeforeSplitting;

// Helper function to split a value (Dot C operand) along a specific axis into
// numSlices. Performs Reshape + Transpose + Split + ConvertLayout.
static SmallVector<Value> splitValueAlongAxis(Value input, int32_t numSlices,
                                              int axis, Location loc,
                                              OpBuilder &builder) {
  if (numSlices == 1) {
    return {input};
  }

  auto splitOnce = [&](Value val) -> std::pair<Value, Value> {
    RankedTensorType inputType = cast<RankedTensorType>(val.getType());
    auto shape = inputType.getShape();
    int rank = shape.size();
    assert(axis < rank);
    // Reshape to split the target axis into <..., 2, N/2, ...>
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape[axis] /= 2;
    newShape.insert(newShape.begin() + axis, 2);
    Value reshaped = triton::ReshapeOp::create(builder, loc, newShape, val);

    // Permute so 2 is last dim <..., 2>
    rank++; // After reshape, we have one more dimension
    SmallVector<int32_t> order;
    for (int i = 0; i < rank; ++i) {
      if (i != axis)
        order.push_back(i);
    }
    order.push_back(axis);
    Value transposed = triton::TransOp::create(builder, loc, reshaped, order);

    // Split along the last dimension.
    triton::SplitOp split = triton::SplitOp::create(builder, loc, transposed);
    Attribute originalEncoding =
        cast<RankedTensorType>(input.getType()).getEncoding();
    SmallVector<Value> converted;
    for (Value result : split.getResults()) {
      auto resultType = cast<RankedTensorType>(result.getType());
      auto targetType = RankedTensorType::get(
          resultType.getShape(), resultType.getElementType(), originalEncoding);
      converted.push_back(triton::gpu::ConvertLayoutOp::create(
          builder, loc, targetType, result));
    }
    return {converted[0], converted[1]};
  };

  // Iteratively split until we reach numSlices
  SmallVector<Value> tiles;
  tiles.push_back(input);
  int32_t currentCount = 1;
  while (currentCount < numSlices) {
    RankedTensorType tileType = cast<RankedTensorType>(tiles[0].getType());
    typesBeforeSplitting.push_back(tileType);
    SmallVector<Value> nextTiles;
    for (Value tile : tiles) {
      auto [left, right] = splitOnce(tile);
      nextTiles.push_back(left);
      nextTiles.push_back(right);
    }
    tiles = std::move(nextTiles);
    currentCount *= 2;
  }
  return tiles;
}

// Helper function (inverse of splitValueAlongAxis) to join Values (D opd of
// Dot) along a specific axis. Performs Join + Transpose + Reshape +
// ConvertLayout.
static Value joinValuesAlongAxis(SmallVector<Value> tiles, int axis,
                                 Location loc, OpBuilder &builder) {
                                  
  auto joinOnce = [&](Value left, Value right,
                      RankedTensorType dstType) -> Value {
    auto leftType = cast<RankedTensorType>(left.getType());
    auto shape = leftType.getShape();
    int rank = shape.size();
    assert(axis < rank);

    // Join creates dim <..., 2>
    Value joined = triton::JoinOp::create(builder, loc, left, right);

    // Transpose to <..., 2, N,...>
    // for axis=0, trans=[2, 0, 1]
    // for axis=1, trans=[0, 2, 1]
    SmallVector<int32_t> trans(rank + 1);
    for (int j = 0; j < rank; ++j) {
      trans[j < axis ? j : j + 1] = j;
    }
    trans[axis] = rank;
    Value transposed = triton::TransOp::create(builder, loc, joined, trans);

    // Reshape to <..., 2N, ...>
    auto transposedType = cast<RankedTensorType>(transposed.getType());
    auto transposedShape = transposedType.getShape();
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape[axis] *= 2;
    Value reshaped =
        triton::ReshapeOp::create(builder, loc, newShape, transposed);

    // Convert back to dst encoding (saved during splits)
    Value converted =
        triton::gpu::ConvertLayoutOp::create(builder, loc, dstType, reshaped);
    return converted;
  };

  // Iteratively join
  while (tiles.size() > 1) {
    RankedTensorType dstType = typesBeforeSplitting.pop_back_val();
    SmallVector<Value> nextTiles;
    for (size_t i = 0; i < tiles.size(); i += 2) {
      Value joined = joinOnce(tiles[i], tiles[i + 1], dstType);
      nextTiles.push_back(joined);
    }
    tiles = std::move(nextTiles);
  }

  return tiles[0];
}

/// Create a new dot or dot_scaled op with the given slice operands.
/// For DotScaledOp, scale slices must be provided via the optional maps.
static Operation *
createDotOp(Operation *dotOp, OpBuilder &builder, Location loc,
            RankedTensorType dType, Value aSlice, Value bSlice, Value cSlice,
            const DenseMap<std::pair<int32_t, int32_t>, Value> *mKToAScale,
            const DenseMap<std::pair<int32_t, int32_t>, Value> *nKToBScale,
            int32_t mOff, int32_t nOff, int32_t kOff) {
  if (auto dot = dyn_cast<triton::DotOp>(dotOp)) {
    return triton::DotOp::create(builder, loc, dType,
                                 ValueRange{aSlice, bSlice, cSlice},
                                 dot->getAttrs());
  }
  if (auto scaledDot = dyn_cast<triton::DotScaledOp>(dotOp)) {
    Value aScale = Value();
    Value bScale = Value();
    if (mKToAScale) {
      auto it = mKToAScale->find({mOff, kOff});
      if (it != mKToAScale->end())
        aScale = it->second;
    }
    if (nKToBScale) {
      auto it = nKToBScale->find({nOff, kOff});
      if (it != nKToBScale->end())
        bScale = it->second;
    }
    return triton::DotScaledOp::create(
        builder, loc, dType, aSlice, bSlice, cSlice, aScale, bScale,
        scaledDot.getAElemType(), scaledDot.getBElemType(),
        scaledDot.getFastMath(), scaledDot.getLhsKPack(),
        scaledDot.getRhsKPack());
  }
  llvm_unreachable("createDotOp: unsupported dot op type");
}

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  /// minimum transpose width
  unsigned minTransposeWidth;

  /// dots to be prefetched (DotOp or DotScaledOp, both implement
  /// DotOpInterface)
  SetVector<Operation *> dots;
  /// dot op => dot operand
  DenseMap<Operation *, Value> dot2aLoopArg;
  DenseMap<Operation *, Value> dot2aHeaderDef;
  DenseMap<Operation *, Value> dot2bLoopArg;
  DenseMap<Operation *, Value> dot2bHeaderDef;
  DenseMap<Operation *, Value> dot2aYield;
  DenseMap<Operation *, Value> dot2bYield;
  DenseMap<Operation *, SmallVector<Value>> dot2aVals;
  DenseMap<Operation *, SmallVector<Value>> dot2bVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  FailureOr<Value> getAsyncWaitTokenForLocalLoad(Operation *cvt,
                                                 bool fromPriorIter,
                                                 OpBuilder &builder,
                                                 IRMapping *mapping = nullptr);

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         std::optional<Value> asyncWaitToken = std::nullopt,
                         std::optional<int64_t> offsetM = std::nullopt,
                         std::optional<int64_t> shapeM = std::nullopt,
                         std::optional<int64_t> offsetN = std::nullopt,
                         std::optional<int64_t> shapeN = std::nullopt,
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

  Operation *generateDotsAndNonPrefetchingLocalLoads(Operation *dotOp,
                                                     Attribute dotEncoding,
                                                     OpBuilder &builder,
                                                     IRMapping &mapping,
                                                     scf::ForOp newForOp);

  void generatePrefetchingLocalLoads(Operation *dotOp, OpBuilder &builder,
                                     IRMapping &mapping,
                                     SmallVector<Value> &yieldValues);

protected:
  /// Prefetch tile dimensions; set by computePrefetchWidths().
  unsigned prefetchWidthM = 0;
  unsigned prefetchWidthN = 0;
  unsigned prefetchWidthK = 0;
  /// Store original kWidth to maintain when creating new local_loads.
  unsigned kWidth = 0;

  unsigned getMinTransposeWidth() const { return minTransposeWidth; }

  /// Target-specific: compute prefetch widths from dot encoding and shapes.
  /// Returns true if widths were set and this dot should be prefetched;
  /// false to skip this dot (e.g. kSize too small).
  virtual bool computePrefetchWidths(Attribute dotEncoding,
                                     unsigned aTypeBitWidth,
                                     ArrayRef<int64_t> dShape, unsigned mSize,
                                     unsigned nSize, unsigned kSize,
                                     unsigned kWidth, bool transA, bool transB,
                                     unsigned minTransposeWidth) = 0;

  /// Target-specific: optionally insert a scheduling barrier (e.g. AMD).
  /// Default is no-op.
  // TODO(dtanner) remove this before upstreaming; AMD-specific code is only
  // experimental/temporary.
  virtual void maybeInsertSchedulingBarrier(OpBuilder &builder, Location loc) {}

public:
  Prefetcher() = delete;
  virtual ~Prefetcher() = default;

  Prefetcher(scf::ForOp forOp, int minTransposeWidth)
      : forOp(forOp), minTransposeWidth(minTransposeWidth) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

void Prefetcher::cloneElementwiseOps(Value &ret, const SmallVector<Value> &vals,
                                     OpBuilder &builder) {
  IRMapping mapping;
  mapping.map(vals[1], ret);
  for (int i = 2; i < vals.size(); i++) {
    Value v = vals[i];
    Value curr = builder.clone(*v.getDefiningOp(), mapping)->getResult(0);
    if (isa<RankedTensorType>(curr.getType())) {
      auto retType = RankedTensorType::get(
          cast<RankedTensorType>(ret.getType()).getShape(),
          cast<RankedTensorType>(curr.getType()).getElementType(),
          cast<RankedTensorType>(curr.getDefiningOp()->getOperand(0).getType())
              .getEncoding());
      curr.setType(retType);
    }
    mapping.map(v, curr);
  }
  if (vals.size() > 1)
    ret = mapping.lookup(vals.back());
}

// Generates all dots and first N-1 local_loads.
// First splits C opd along M and N, then loop over M, N, K creating dot
// sub-tiles and local_loads, and finally joins D opds along M and N.
Operation *Prefetcher::generateDotsAndNonPrefetchingLocalLoads(
    Operation *dotOp, Attribute dotEncoding, OpBuilder &builder,
    IRMapping &mapping, scf::ForOp newForOp) {
  auto dotInterface = cast<triton::DotOpInterface>(dotOp);
  // Get total dimensions from operands
  auto aType = cast<RankedTensorType>(dotInterface.getA().getType());
  auto bType = cast<RankedTensorType>(dotInterface.getB().getType());
  int64_t totalM = aType.getShape()[0];
  int64_t totalK = aType.getShape().back();
  int64_t totalN = bType.getShape().back();
  Location loc = dotOp->getLoc();

  // Map from (M, N) offsets to dot C/D opds
  DenseMap<std::pair<int32_t, int32_t>, Value> mnToDot;

  // Assert that dimensions are evenly divisible by prefetch widths
  assert(totalM % prefetchWidthM == 0 &&
         "totalM must be divisible by prefetchWidthM");
  assert(totalN % prefetchWidthN == 0 &&
         "totalN must be divisible by prefetchWidthN");
  assert(totalK % prefetchWidthK == 0 &&
         "totalK must be divisible by prefetchWidthK");

  // Slice c opd along M
  Value cOperand = mapping.lookup(dotOp->getOperand(2));
  int mAxis = 0;
  int nAxis = 1;
  int32_t numSlicesM = totalM / prefetchWidthM;
  SmallVector<Value> mSlices =
      splitValueAlongAxis(cOperand, numSlicesM, mAxis, loc, builder);
  // Slice c opds along N
  int32_t numSlicesN = totalN / prefetchWidthN;
  for (int32_t mIdx = 0; mIdx < numSlicesM; ++mIdx) {
    int32_t mOff = mIdx * prefetchWidthM;
    SmallVector<Value> mnSlices =
        splitValueAlongAxis(mSlices[mIdx], numSlicesN, nAxis, loc, builder);
    for (int32_t nIdx = 0; nIdx < numSlicesN; ++nIdx) {
      int32_t nOff = nIdx * prefetchWidthN;
      mnToDot[{mOff, nOff}] = mnSlices[nIdx];
    }
  }

  // For DotScaledOp, build (mOff,kOff) -> a_scale slice and (nOff,kOff) ->
  // b_scale slice
  std::optional<DenseMap<std::pair<int32_t, int32_t>, Value>> mKToAScale;
  std::optional<DenseMap<std::pair<int32_t, int32_t>, Value>> nKToBScale;
  if (auto scaledDot = dyn_cast<triton::DotScaledOp>(dotOp)) {
    int32_t numSlicesK = totalK / prefetchWidthK;
    int scaleFactor = 32;
    if (Value aScaleVal = scaledDot.getAScale()) {
      auto scaleTy = cast<RankedTensorType>(aScaleVal.getType());
      if (isa<mlir::Float8E4M3FNType>(scaleTy.getElementType()))
        scaleFactor = 16;
      mKToAScale.emplace();
      Value aScaleMapped = mapping.lookup(aScaleVal);
      SmallVector<Value> aScaleMSlices =
          splitValueAlongAxis(aScaleMapped, numSlicesM, 0, loc, builder);
      for (int32_t mIdx = 0; mIdx < numSlicesM; ++mIdx) {
        int32_t mOff = mIdx * prefetchWidthM;
        SmallVector<Value> aScaleKSlices = splitValueAlongAxis(
            aScaleMSlices[mIdx], numSlicesK, 1, loc, builder);
        for (int32_t kIdx = 0; kIdx < numSlicesK; ++kIdx) {
          int32_t kOff = kIdx * prefetchWidthK;
          (*mKToAScale)[{mOff, kOff}] = aScaleKSlices[kIdx];
        }
      }
    }
    if (Value bScaleVal = scaledDot.getBScale()) {
      auto scaleTy = cast<RankedTensorType>(bScaleVal.getType());
      if (isa<mlir::Float8E4M3FNType>(scaleTy.getElementType()))
        scaleFactor = 16;
      nKToBScale.emplace();
      Value bScaleMapped = mapping.lookup(bScaleVal);
      SmallVector<Value> bScaleNSlices =
          splitValueAlongAxis(bScaleMapped, numSlicesN, 0, loc, builder);
      for (int32_t nIdx = 0; nIdx < numSlicesN; ++nIdx) {
        int32_t nOff = nIdx * prefetchWidthN;
        SmallVector<Value> bScaleKSlices = splitValueAlongAxis(
            bScaleNSlices[nIdx], numSlicesK, 1, loc, builder);
        for (int32_t kIdx = 0; kIdx < numSlicesK; ++kIdx) {
          int32_t kOff = kIdx * prefetchWidthK;
          (*nKToBScale)[{nOff, kOff}] = bScaleKSlices[kIdx];
        }
      }
    }
    (void)scaleFactor;
  }

  // Generate dots[m, n, k] and local_loads[m, n, k] (except for local_load[0,
  // 0, 0] which is prefetched) Insertion point is manipulated to ensure
  // ordering of local_load[x+1] before dot[x]
  Operation *lastDotOp = nullptr;
  for (int32_t kOff = 0; kOff < totalK; kOff += prefetchWidthK) {
    // Store local loads, since one loaded opd is reused for multiple dots.
    DenseMap<int32_t, Value> aSlices;
    DenseMap<int32_t, Value> bSlices;
    for (int32_t mOff = 0; mOff < totalM; mOff += prefetchWidthM) {
      for (int32_t nOff = 0; nOff < totalN; nOff += prefetchWidthN) {
        if (lastDotOp)
          builder.setInsertionPoint(lastDotOp);

        Value aSlice; // used for dot creation
        if (kOff == 0 && mOff == 0) {
          // Opd was prefetched in prior kernel loop iter.
          Value a = operand2headPrefetch[dotInterface.getA()];
          aSlice = newForOp.getTiedLoopRegionIterArg(&*a.use_begin());
          aSlices[mOff] = aSlice;
        } else {
          if (nOff == 0) {
            // Create new load for kOff>0 and nOff=0.
            FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
                dot2aVals[dotOp].back().getDefiningOp(), false, builder,
                &mapping);
            aSlice = generatePrefetch(
                mapping.lookup(dot2aLoopArg[dotOp]), 0, false, dotEncoding,
                builder,
                failed(awtA) ? std::nullopt : std::optional<Value>(*awtA), mOff,
                prefetchWidthM, std::nullopt, std::nullopt, kOff,
                prefetchWidthK);
            cloneElementwiseOps(aSlice, dot2aVals[dotOp], builder);
            aSlices[mOff] = aSlice;
          } else {
            // Reuse the opd previously created during nOff=0 for nOff>0.
            aSlice = aSlices[mOff];
          }
        }

        Value bSlice; // used for dot creation
        if (kOff == 0 && nOff == 0) {
          // Opd was prefetched in prior iter.
          Value b = operand2headPrefetch[dotInterface.getB()];
          bSlice = newForOp.getTiedLoopRegionIterArg(&*b.use_begin());
          bSlices[nOff] = bSlice;
        } else {
          if (mOff == 0) {
            // Create new local load for kOff>0 and mOff=0.
            FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
                dot2bVals[dotOp].back().getDefiningOp(), false, builder,
                &mapping);
            bSlice = generatePrefetch(
                mapping.lookup(dot2bLoopArg[dotOp]), 1, false, dotEncoding,
                builder,
                failed(awtB) ? std::nullopt : std::optional<Value>(*awtB),
                std::nullopt, std::nullopt, nOff, prefetchWidthN, kOff,
                prefetchWidthK);
            cloneElementwiseOps(bSlice, dot2bVals[dotOp], builder);
            bSlices[nOff] = bSlice;
          } else {
            // Reuse the opd previously created during mOff=0 for mOff>0.
            bSlice = bSlices[nOff];
          }
        }

        if (lastDotOp)
          builder.setInsertionPointAfter(lastDotOp);
        maybeInsertSchedulingBarrier(builder, loc);
        Value cSlice = mnToDot[{mOff, nOff}];
        auto dType = cast<RankedTensorType>(cSlice.getType());
        const DenseMap<std::pair<int32_t, int32_t>, Value> *aScaleMap =
            mKToAScale ? &*mKToAScale : nullptr;
        const DenseMap<std::pair<int32_t, int32_t>, Value> *bScaleMap =
            nKToBScale ? &*nKToBScale : nullptr;
        Operation *newDot =
            createDotOp(dotOp, builder, loc, dType, aSlice, bSlice, cSlice,
                        aScaleMap, bScaleMap, mOff, nOff, kOff);
        mnToDot[{mOff, nOff}] = newDot->getResult(0);
        lastDotOp = newDot;
      }
    }
  }

  // Concatenate all M×N tiles back into a single tensor with original shape
  // Join d opds along N
  SmallVector<Value> mJoins;
  for (int32_t mOff = 0; mOff < totalM; mOff += prefetchWidthM) {
    SmallVector<Value> mnSlices;
    for (int32_t nOff = 0; nOff < totalN; nOff += prefetchWidthN) {
      mnSlices.push_back(mnToDot[{mOff, nOff}]);
    }
    Value mJoin = joinValuesAlongAxis(mnSlices, nAxis, loc, builder);
    mJoins.push_back(mJoin);
  }
  // Join d opds along M
  Value result = joinValuesAlongAxis(mJoins, mAxis, loc, builder);
  Operation *newOp = result.getDefiningOp();
  // Reset insertion point to before the last dot for the prefetched local loads
  builder.setInsertionPoint(lastDotOp);
  return newOp;
}

// Generates the prefetched local loads which are for dot[m=0,n=0,k=0]
void Prefetcher::generatePrefetchingLocalLoads(
    Operation *dotOp, OpBuilder &builder, IRMapping &mapping,
    SmallVector<Value> &yieldValues) {
  Attribute dotEncoding =
      cast<RankedTensorType>(dotOp->getResult(0).getType()).getEncoding();
  // Get async wait tokens from async_wait at end of prior iteration.
  FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
      dot2aVals[dotOp].back().getDefiningOp(), true, builder, &mapping);
  FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
      dot2bVals[dotOp].back().getDefiningOp(), true, builder, &mapping);
  Value aToYield = generatePrefetch(
      mapping.lookup(dot2aYield[dotOp]), 0, true, dotEncoding, builder,
      failed(awtA) ? std::nullopt : std::optional<Value>(*awtA));
  cloneElementwiseOps(aToYield, dot2aVals[dotOp], builder);
  yieldValues.push_back(aToYield);
  Value bToYield = generatePrefetch(
      mapping.lookup(dot2bYield[dotOp]), 1, true, dotEncoding, builder,
      failed(awtB) ? std::nullopt : std::optional<Value>(*awtB));
  cloneElementwiseOps(bToYield, dot2bVals[dotOp], builder);
  yieldValues.push_back(bToYield);
}

// Get async wait token (awt), if any, for new LocalLoad in newForOp
// based on old LocalLoad; args determine 3 cases where to
// get/create awt.
//
// Args
// - fromPriorIter, used for prefetching slice[0], means track the awt
//   through block args, yield and find it in the previous loop iteration.
// - mapping maps original forOp to newForOp, and is not used with
//   not in for loop, e.g. for emitPrologue.
//
// Case 0 - Prologue. awt is loop arg; returns init value before loop.
//  - fromPriorIter=false
//  - mapping=nullptr
// Case 1 - Slice[1,N-1]. awt is loop arg; returns same arg but mapped to
// newForLoop.
//  - fromPriorIter=false
//  - mapping=valid
// Case 2 - Slice[0] prefetched. awt comes from end of prior loop iteration.
//  - fromPriorIter=true
//  - mapping=valid
//
//  NOTE: fromPriorIter=true & mapping=nullptr is invalid combination.
FailureOr<Value> Prefetcher::getAsyncWaitTokenForLocalLoad(Operation *cvt,
                                                           bool fromPriorIter,
                                                           OpBuilder &builder,
                                                           IRMapping *mapping) {
  auto llOp = dyn_cast<triton::gpu::LocalLoadOp>(cvt);
  if (!llOp)
    return failure();
  if (llOp->getNumOperands() != 2)
    return failure();
  Value awt = llOp->getOperand(1);
  if (!isa<AsyncTokenType>(awt.getType()))
    return failure();

  if (!fromPriorIter) {
    if (!mapping) {
      // Case 0: return async wait token in prologue.
      if (mlir::BlockArgument loopArg = dyn_cast<mlir::BlockArgument>(awt)) {
        unsigned argIdx = loopArg.getArgNumber() - forOp.getNumInductionVars();
        Value initAwt = forOp.getInitArgs()[argIdx];
        return initAwt;
      } else {
        assert(false || "Expected async wait token to be loop arg.");
        return failure();
      }
      return awt;
    } else {
      // Case 1: return new async wait token from for(args) for
      // LocalLoad[1, N-1].
      return mapping->lookup(awt);
    }
  }
  assert(mapping);
  assert(fromPriorIter);

  mlir::BlockArgument loopArg = dyn_cast<mlir::BlockArgument>(awt);
  if (!loopArg) {
    assert(false || "fromPriorIter specified but awt isn't a loop arg.");
    return failure();
  }

  // Case 2: return new async wait token from end of prior iteration,
  // this occurs for the prefetching LocalLoads at the end of the loop;
  // which may or may not have been created yet i.e. is in mapping.
  // Note: awt may already be in mapping for two reasons,
  // (a) it is a duplicate of async_wait created below,
  // (b) associated async_wait was already created previously in new loop
  // even though want prior iter of it. Now we want to wrap around the loop
  // body and find this token in the previous iteration because it was
  // prefetched.
  unsigned argIdx = loopArg.getArgNumber() - forOp.getNumInductionVars();
  Value initAwt = forOp.getInitArgs()[argIdx];
  Value yieldedAwt = yieldOp.getOperand(argIdx);
  if (mapping->contains(yieldedAwt))
    return mapping->lookup(yieldedAwt);

  // Want awt fromPriorIter, but it isn't in map yet because the async_wait op
  // hasn't been visited yet, so create and place in mapping.
  LDBG("Case 2 yieldedAwt not yet in map");
  auto awOp = yieldedAwt.getDefiningOp();
  // Create new async_wait op in new loop
  Operation *newAwOp = builder.clone(*awOp, *mapping);
  for (unsigned dstIdx : llvm::seq(unsigned(0), awOp->getNumResults()))
    mapping->map(awOp->getResult(dstIdx), newAwOp->getResult(dstIdx));
  return newAwOp->getResult(0);
}

// Since dots have 3D slicing, the MemDescSubslice for loca loads
// will have 2D offsets and shapes.
Value Prefetcher::generatePrefetch(
    Value v, unsigned opIdx, bool isPrologue, Attribute dotEncoding,
    OpBuilder &builder, std::optional<Value> asyncWaitToken,
    std::optional<int64_t> offsetM, std::optional<int64_t> shapeM,
    std::optional<int64_t> offsetN, std::optional<int64_t> shapeN,
    std::optional<int64_t> offsetK, std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int32_t> offset(rank, 0);
  Type elementType = type.getElementType();

  // For operand A (opIdx=0): shape is [M, K], so mIdx=0, kIdx=1
  // For operand B (opIdx=1): shape is [K, N], so kIdx=0, nIdx=1
  int64_t mIdx = 0; // M dimension index (only for operand A)
  int64_t nIdx = 1; // N dimension index (only for operand B)
  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  // Handle m dim for opd A
  if (opIdx == 0) {
    offset[mIdx] = isPrologue ? 0 : prefetchWidthM;
    shape[mIdx] = isPrologue ? prefetchWidthM : (shape[mIdx] - prefetchWidthM);
    if (shapeM)
      shape[mIdx] = *shapeM;
    if (offsetM)
      offset[mIdx] = *offsetM;
  }

  // Handle n dim for opd B
  if (opIdx == 1) {
    offset[nIdx] = isPrologue ? 0 : prefetchWidthN;
    shape[nIdx] = isPrologue ? prefetchWidthN : (shape[nIdx] - prefetchWidthN);
    if (shapeN)
      shape[nIdx] = *shapeN;
    if (offsetN)
      offset[nIdx] = *offsetN;
  }

  // Handle k dim
  offset[kIdx] = isPrologue ? 0 : prefetchWidthK;
  shape[kIdx] = isPrologue ? prefetchWidthK : (shape[kIdx] - prefetchWidthK);
  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  Value newSmem = triton::gpu::MemDescSubsliceOp::create(
      builder, v.getLoc(),
      triton::gpu::MemDescType::get(
          shape, elementType, type.getEncoding(), type.getMemorySpace(),
          type.getMutableMemory(), type.getAllocShape()),
      v, offset);
  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, kWidth);
  Value prefetchSlice = triton::gpu::LocalLoadOp::create(
      builder, v.getLoc(),
      RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem,
      asyncWaitToken.value_or(nullptr));
  return prefetchSlice;
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<Operation *> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotInterface = dyn_cast<triton::DotOpInterface>(&op)) {
      // Only accepts dot ops encoded as Nvidia MMA v2 or AMD MFMA/WMMA
      Value result = dotInterface.getD();
      auto dstMmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(result));
      auto dstMfmaEnc = dyn_cast<AMDMfmaEncodingAttr>(getEncoding(result));
      auto dstWmmaEnc = dyn_cast<AMDWmmaEncodingAttr>(getEncoding(result));
      if (!dstMfmaEnc && (!dstMmaEnc || dstMmaEnc.getVersionMajor() != 2) &&
          !dstWmmaEnc)
        // Don't rewrite if any other type is found.
        return failure();
      dotsInFor.push_back(&op);
    }

  if (dotsInFor.empty())
    return failure();

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  // returns source of cvt
  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // walk back to conversion
    Operation *op = v.getDefiningOp();
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    LDBG("Prefetch src: " << *op);
    while (op) {
      if (!op->getResult(0).hasOneUse())
        break;
      rets.push_back(op->getOperand(0));
      if (auto cvt = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        // NYI for other encodings, for example if we have transpose
        // in the chain
        if (isa<DotOperandEncodingAttr>(cvt.getType().getEncoding()))
          foundConvertFromShared = true;
        break;
      }
      op = op->getOperand(0).getDefiningOp();
      if (op)
        LDBG("op: " << *op);
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = mlir::dyn_cast<BlockArgument>(v))
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getTiedLoopInit(arg)->get();
    return Value();
  };

  auto getYieldOperand = [this](Value v) -> Value {
    auto arg = mlir::cast<BlockArgument>(v);
    unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    return yieldOp.getOperand(yieldIdx);
  };

  for (Operation *dotOp : dotsInFor) {
    auto dotInterface = cast<triton::DotOpInterface>(dotOp);
    auto aOpd = dotInterface.getA();
    auto bOpd = dotInterface.getB();
    auto aType = cast<RankedTensorType>(aOpd.getType());
    auto bType = cast<RankedTensorType>(bOpd.getType());
    auto dType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    auto aEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(bType.getEncoding());
    assert(aEnc.getKWidth() == bEnc.getKWidth());
    kWidth = aEnc.getKWidth();
    LDBG("kWidth: " << kWidth);

    auto transOp = [&](Operation *op, int opdIdx) -> bool {
      if (auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        auto srcType = localLoad.getSrc().getType();
        auto order = getOrder(srcType);
        return (order[0] == opdIdx);
      }
      return true;
    };

    // Get sizes for all three dimensions
    unsigned mSize = aType.getShape()[0];     // M dimension from operand A
    unsigned nSize = bType.getShape().back(); // N dimension from operand B
    unsigned kSize = aType.getShape().back(); // K dimension

    bool transA = transOp(aOpd.getDefiningOp(), 0);
    bool transB = transOp(bOpd.getDefiningOp(), 1);
    Attribute dotEncoding =
        cast<RankedTensorType>(dotOp->getResult(0).getType()).getEncoding();
    if (!computePrefetchWidths(dotEncoding, aType.getElementTypeBitWidth(),
                               dType.getShape(), mSize, nSize, kSize, kWidth,
                               transA, transB, getMinTransposeWidth()))
      continue;
    LDBG("prefetchWidthMNK: " << prefetchWidthM << "x" << prefetchWidthN << "x"
                              << prefetchWidthK);
    assert(prefetchWidthM > 0 && prefetchWidthN > 0 && prefetchWidthK > 0);
    auto aVals = getPrefetchSrc(dotInterface.getA());
    auto bVals = getPrefetchSrc(dotInterface.getB());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      Value aHeaderDef = getIncomingOp(aSmem);
      Value bHeaderDef = getIncomingOp(bSmem);
      // Only prefetch loop arg
      if (aHeaderDef && bHeaderDef) {
        dots.insert(dotOp);
        dot2aVals[dotOp] = aVals;
        dot2bVals[dotOp] = bVals;
        dot2aHeaderDef[dotOp] = aHeaderDef;
        dot2bHeaderDef[dotOp] = bHeaderDef;
        dot2aLoopArg[dotOp] = aSmem;
        dot2bLoopArg[dotOp] = bSmem;
        dot2aYield[dotOp] = getYieldOperand(aSmem);
        dot2bYield[dotOp] = getYieldOperand(bSmem);
      }
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (Operation *dotOp : dots) {
    auto dotInterface = cast<triton::DotOpInterface>(dotOp);
    FailureOr<Value> awtA = getAsyncWaitTokenForLocalLoad(
        dot2aVals[dotOp].back().getDefiningOp(), false, builder);
    FailureOr<Value> awtB = getAsyncWaitTokenForLocalLoad(
        dot2bVals[dotOp].back().getDefiningOp(), false, builder);
    Attribute dotEncoding =
        cast<RankedTensorType>(dotOp->getResult(0).getType()).getEncoding();
    Value aPrefetched = generatePrefetch(
        dot2aHeaderDef[dotOp], 0, true, dotEncoding, builder,
        failed(awtA) ? std::nullopt : std::optional<Value>(*awtA));
    cloneElementwiseOps(aPrefetched, dot2aVals[dotOp], builder);
    Value bPrefetched = generatePrefetch(
        dot2bHeaderDef[dotOp], 1, true, dotEncoding, builder,
        failed(awtB) ? std::nullopt : std::optional<Value>(*awtB));
    cloneElementwiseOps(bPrefetched, dot2bVals[dotOp], builder);
    operand2headPrefetch[dotInterface.getA()] = aPrefetched;
    operand2headPrefetch[dotInterface.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (Operation *dotOp : dots) {
    auto dotInterface = cast<triton::DotOpInterface>(dotOp);
    loopArgs.push_back(operand2headPrefetch[dotInterface.getA()]);
    loopArgs.push_back(operand2headPrefetch[dotInterface.getB()]);
  }

  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  // The insertion point should be placed before the yield op
  auto setInsertionPointBeforeYield = [](OpBuilder &builder,
                                         scf::ForOp newForOp) {
    if (newForOp.getBody()->mightHaveTerminator()) {
      builder.setInsertionPoint(newForOp.getBody()->getTerminator());
    } else {
      builder.setInsertionPointToEnd(newForOp.getBody());
    }
  };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    // If we're currently trying to sink a prefetched dot, we need to stop
    // sinking it (by resetting the insertion point to the end) if we find
    // control flow, or anything that depends on the dot op.
    if (op.getNumRegions() > 0) {
      setInsertionPointBeforeYield(builder, newForOp);
    }
    for (auto operand : op.getOperands()) {
      if (auto def = operand.getDefiningOp()) {
        if (dots.contains(def)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    if (dots.contains(&op)) {
      Attribute dotEncoding =
          cast<RankedTensorType>(op.getResult(0).getType()).getEncoding();
      newOp = generateDotsAndNonPrefetchingLocalLoads(&op, dotEncoding, builder,
                                                      mapping, newForOp);
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (Operation *dotOp : dots) {
    generatePrefetchingLocalLoads(dotOp, builder, mapping, yieldValues);
  }
  // Update ops of yield
  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    scf::YieldOp::create(builder, yieldOp.getLoc(), yieldValues);
  return newForOp;
}

/*
  AMD-specific prefetch is based on prefetching enough data early enough
  to not be stalled by lds latency, but not prefetch too much as this
  consumes excessive vgprs, and reduces cycles which can be used for
  HBM-latency hiding.

  The degree of prefetching is expressed as numInsts,
  which is the number of mfma instructions to prefetch lds data by.
  It is tuned from the lds latency cycles and the mfma cycles.
  E.g. if max lds latency is 40 cycles, and mfma is 16 cycles,
  then we choose 40/16 = 2.5 -> 4 (to get next power of 2).
  This will ensure there are 4 mfma in one sliced dot,
  and in the llvm, there will be 4 mfmas between a local_load
  and the mfma which depends on it.

  Therefore higher lds latency, and faster mfma instructions
  leads to larger numInsts -> larger prefetchWidths,
  fewer splits/joins, and we prefetch more data which uses more vgprs.

  numInsts is fed into prefetchWidth() to determine prefetchWidthMNK
  which specifies the tile shape of the sliced dots.
  The logic for this tile shape is
  - If tranposing A or B (e.g. ds_read_tr), then K and M or N are not sliced
    in order to preserve a minimum transpose width.
  - Whether the MxN shape prefers to be square or row; e.g. for numInst=4,
    square => 2x2 while row => 4x1. Square vs row impacts
    (a) Number of local_loads ops needed to prefetch data for first 4 mfmas.
    (b) How local_loads are distributed during the series of mfmas.
    (c) More research can be done here.
*/
class PrefetchAMD : public Prefetcher {
  using Prefetcher::Prefetcher;

  static std::tuple<unsigned, unsigned, unsigned>
  prefetchWidth(unsigned mSize, unsigned nSize, unsigned kSize,
                unsigned minTransposeWidth, bool transA, bool transB,
                ArrayRef<unsigned> instrShape, ArrayRef<unsigned> warpsPerCta,
                unsigned numInsts) {
    LDBG("instrShape: " << instrShape[0] << "x" << instrShape[1] << "x"
                        << instrShape[2]);
    LDBG("warpsPerCta: " << warpsPerCta[0] << "x" << warpsPerCta[1]);
    LDBG("TotalInsts: " << mSize / (instrShape[0] * warpsPerCta[0]) << "x"
                        << nSize / (instrShape[1] * warpsPerCta[1]) << "x"
                        << kSize / instrShape[2] << " (" << numInsts << ")");
    // mnk specify num ops a sliced dot
    unsigned m = 1, n = 1, k = 1;
    unsigned maxM = mSize / (instrShape[0] * warpsPerCta[0]);
    unsigned maxN = nSize / (instrShape[1] * warpsPerCta[1]);
    unsigned maxK = kSize / (instrShape[2]);
    if (transA) {
      m = std::max<unsigned>(m, minTransposeWidth / instrShape[0]);
      k = std::max<unsigned>(k, minTransposeWidth / instrShape[2]);
    }
    if (transB) {
      n = std::max<unsigned>(n, minTransposeWidth / instrShape[1]);
      k = std::max<unsigned>(k, minTransposeWidth / instrShape[2]);
    }
    numInsts /= (m * n * k);
    LDBG("instr tile m: " << m << ", n: " << n << ", k: " << k);
    // Iteratively increase the tile shape until we reach numInsts
    // according to the preferred shape.
    // Currently, LLVM scheduling seems to schedule rows better than squares.
    bool preferSquare = false;
    while (numInsts > 1) {

      if ((m <= n || !preferSquare) && m < maxM && !transA) {
        m *= 2;
      } else if (n < maxN) {
        n *= 2;
      } else if (k < maxK) {
        k *= 2;
      } else {
        break;
      }
      numInsts /= 2;
    }
    LDBG("instr tile m: " << m << ", n: " << n << ", k: " << k);
    // convert num ops to CTA tile shape
    m *= instrShape[0] * warpsPerCta[0];
    n *= instrShape[1] * warpsPerCta[1];
    k *= instrShape[2];
    m = std::min<unsigned>(m, mSize);
    n = std::min<unsigned>(n, nSize);
    k = std::min<unsigned>(k, kSize);
    return {m, n, k};
  }

  bool computePrefetchWidths(Attribute dotEncoding, unsigned aTypeBitWidth,
                             ArrayRef<int64_t> dShape, unsigned mSize,
                             unsigned nSize, unsigned kSize, unsigned kWidth,
                             bool transA, bool transB,
                             unsigned minTransposeWidth) {
    if (auto mfmaEnc = dyn_cast<AMDMfmaEncodingAttr>(dotEncoding)) {

      unsigned numInsts = 4;
      std::tie(prefetchWidthM, prefetchWidthN, prefetchWidthK) = prefetchWidth(
          mSize, nSize, kSize, minTransposeWidth, transA, transB,
          mfmaEnc.getInstrShape(), mfmaEnc.getWarpsPerCTA(), numInsts);
      return true;
    }

    if (auto wmmaEnc = dyn_cast<AMDWmmaEncodingAttr>(dotEncoding)) {
      unsigned numInsts = 8;
      auto warpsPerCTA = getWarpsPerCTA(wmmaEnc, dShape);
      std::tie(prefetchWidthM, prefetchWidthN, prefetchWidthK) =
          prefetchWidth(mSize, nSize, kSize, minTransposeWidth, transA, transB,
                        wmmaEnc.getInstrShape(), warpsPerCTA, numInsts);
      return true;
    }
    return false;
  }

  void maybeInsertSchedulingBarrier(OpBuilder &builder, Location loc) {
    // Bitmask that encodes instruction types for LLVM AMD scheduling hints.
    enum InstructionKindMask {
      NONE = 0x0000,
      ALL_ALU = 0x0001,
      VALU = 0x0002,
      SALU = 0x0004,
      MFMA = 0x0008,
      ALL_VMEM = 0x0010,
      VMEM_READ = 0x0020,
      VMEM_WRITE = 0x0040,
      ALL_DS = 0x0080,
      DS_READ = 0x0100,
      DS_WRITE = 0x0200,
      TRANSCEND = 0x0400
    };
    int32_t mask =
        0 | InstructionKindMask::VALU | InstructionKindMask::SALU |
        InstructionKindMask::ALL_VMEM | InstructionKindMask::VMEM_READ |
        InstructionKindMask::VMEM_WRITE | InstructionKindMask::TRANSCEND;
    if (tools::getBoolEnv("TRITON_HIP_PREFETCH_INSERT_SCHED_BARRIER")) {
      ROCDL::SchedBarrier::create(builder, loc, mask);
    }
  }
};

/// NVIDIA-specific prefetch: MMA encoding.
class PrefetchNV : public Prefetcher {
  using Prefetcher::Prefetcher;
  bool computePrefetchWidths(Attribute dotEncoding, unsigned aTypeBitWidth,
                             ArrayRef<int64_t> dShape, unsigned mSize,
                             unsigned nSize, unsigned kSize, unsigned kWidth,
                             bool transA, bool transB,
                             unsigned minTransposeWidth) {
    auto mmaEnc = dyn_cast<NvidiaMmaEncodingAttr>(dotEncoding);
    if (!mmaEnc)
      return false;
    if (kWidth == 0)
      prefetchWidthK = 256 / aTypeBitWidth;
    else
      prefetchWidthK = 8 * kWidth;
    prefetchWidthM = mSize;
    prefetchWidthN = nSize;
    if (kSize < prefetchWidthK)
      return false;
    return true;
  }
};

} // anonymous namespace

struct PrefetchPass : public impl::TritonGPUPrefetchBase<PrefetchPass> {
  using Base::Base;

  void runOnOperation() override {
    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    ModuleOp module = cast<ModuleOp>(getOperation());
    std::optional<StringRef> arch = getAMDArch(module);
    getOperation()->walk([&](scf::ForOp forOp) {
      std::unique_ptr<Prefetcher> prefetcher;
      if (arch) {
        prefetcher = std::make_unique<PrefetchAMD>(forOp, minTransposeWidth);
      } else {
        prefetcher = std::make_unique<PrefetchNV>(forOp, minTransposeWidth);
      }
      if (prefetcher->initialize().failed())
        return;

      prefetcher->emitPrologue();

      scf::ForOp newForOp = prefetcher->createNewForOp();

      // replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
