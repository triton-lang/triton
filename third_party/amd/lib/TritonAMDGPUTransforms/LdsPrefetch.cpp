//===----------------------------------------------------------------------===//
//
// This pass tries to prefetch operands (a and b) of tt.dot to avoid
// stalling on LDS load latency.
//
// Example transformation:
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
// It is assumed that the loop body has already been pipelined
// for loading data into LDS; now the data will be prefetched from
// lds to improve performance in two ways.
// (1) Rather than a pipelined for loop starting with local_loads
// and incurring an LDS latency at the top of the loop, we prefetch the first
// few local loads at the end of the previous iteration, so that the first op of
// the for loop can be a wmma (ideally).
// (2) Rather than the for loop
// structure being
// %a = ttg.local_load
// %b = ttg.local_load %d = tt.dot %a, %b, %c
// which lowers to llvm with all loads preceeding all wmmas, we prefetch the
// lds data for subtiles which naturally interleaves the local_loads and wmmas.
// Which looks like (for 2x2 slicing along M and N)
// scf.for ... iter_args(%a0, %b0, ...) {
// %b1 = ttg.local_load
// %d00 = tt.dot %a0, %b0, %c00
// %a1 = ttg.local_load
// %d01 = tt.dot %a0, %b1, %c01
// %d10 = tt.dot %a1, %b0, %c10
// %a0' = ttg.local_load
// %b0' = ttg.local_load
// %d11 = tt.dot %a1, %b1, %c11
// scf.yield %a0', %b0', ...
// }
// The improved llvm-ir interleaving can improve backend scheduling.
//
// To control the degree of prefetching, the dots are sliced along M, N, K
// according to prefetchWidthM,N,K. Because the dot is 3D sliced,
// many sliced dots and sliced local_loads are created,
// and the they are emitted such that there is 1 sliced dot between
// local_loads and the dot which depends on them; data for the 0th dot is moved
// to the end of the previous iteration. E.g.
//
// %a1 = ttg.local_load %a_tmp
// %d0 = tt.dot %a0, %b0, %c0 <-- hides ds_read latency
// %d1 = tt.dot %a1, %b1, %c1 <-- %a1 waited for here
//
// Prefetching *all* the A, B operands of dots from local memory would be
// prohibitively expensive in terms of registers and cycles.
// This pass supports slicing along K (which doesn't change the D, C operands
// of the dot) and slicing along M and N.
//
// Currently, prefetching has the following restrictions
// - Only a single dot per loop is supported; relaxing this will rely on have
// the local_loads of one dot being placed inside another sliced dot.
// - Intermediate ops between the dot and local_load are not supported;
// improving this requires being able to identify which ops (e.g. reshape,
// trans) support slicing and to what degree. Then need to prefetch the
// intermediate ops too.
// - DotScaled (with scales) is not supported; improving this requires creating
// new local_loads for scales which behave differently than for operands.
// - Ops which are prefetched at the end of the previous iteration must not
// have any dependencies in the loop body; all dependencies must either be
// loop-carried iter_args or "promotable" expressions (elementwise,
// constant-like, async_wait, or memdesc_index on a ring buffer) whose
// operands are themselves promotable. Promotable expressions are cloned into
// the prologue (with loop-carried args remapped to init values and the
// induction var to the lower bound) and at the end of each iteration (with
// iter_args remapped to their yielded values and the induction var advanced).
// This allows loops with an async_wait + memdesc_index at the top of the loop
// to be prefetched without requiring the user to hoist the async_wait/index
// chain into a loop-carried arg.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PrefetchUtils.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <tuple>

#define DEBUG_TYPE "tritonamdgpu-lds-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPULDSPREFETCH
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace ttg = triton::gpu;

namespace triton {
namespace amdgpu {

namespace {

// Helper function to split a value (Dot C operand) along a specific axis into
// numSlices. Performs Reshape + Transpose + Split + ConvertLayout.
SmallVector<Value>
splitValueAlongAxis(Value input, int32_t numSlices, int axis,
                    SmallVector<RankedTensorType> &typesBeforeSplitting,
                    Location loc, OpBuilder &builder) {
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
      converted.push_back(
          ttg::ConvertLayoutOp::create(builder, loc, targetType, result));
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
static Value
joinValuesAlongAxis(SmallVector<Value> tiles, int axis,
                    SmallVector<RankedTensorType> &typesBeforeSplitting,
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
    SmallVector<int64_t> newShape(shape.begin(), shape.end());
    newShape[axis] *= 2;
    Value reshaped =
        triton::ReshapeOp::create(builder, loc, newShape, transposed);

    // Convert back to dst encoding (saved during splits)
    Value converted =
        ttg::ConvertLayoutOp::create(builder, loc, dstType, reshaped);
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
/// For DotScaledOp, scale operands are looked up from the mapping.
static Operation *createDotOp(Operation *dotOp, OpBuilder &builder,
                              Location loc, RankedTensorType dType,
                              Value aSlice, Value bSlice, Value cSlice,
                              IRMapping *mapping) {
  if (auto dot = dyn_cast<triton::DotOp>(dotOp)) {
    return triton::DotOp::create(builder, loc, dType,
                                 ValueRange{aSlice, bSlice, cSlice},
                                 dot->getAttrs());
  }
  if (auto scaledDot = dyn_cast<triton::DotScaledOp>(dotOp)) {
    Value aScale;
    Value bScale;
    if (Value v = scaledDot.getAScale())
      aScale = mapping->lookup(v);
    if (Value v = scaledDot.getBScale())
      bScale = mapping->lookup(v);
    return triton::DotScaledOp::create(
        builder, loc, dType, aSlice, bSlice, cSlice, aScale, bScale,
        scaledDot.getAElemType(), scaledDot.getBElemType(),
        scaledDot.getFastMath(), scaledDot.getLhsKPack(),
        scaledDot.getRhsKPack());
  }
  llvm_unreachable("createDotOp: unsupported dot op type");
}

class Prefetcher {
public:
  Prefetcher() = delete;
  ~Prefetcher() = default;

  Prefetcher(scf::ForOp forOp, unsigned numInstsOverride = 0)
      : forOp(forOp), numInstsOverride(numInstsOverride) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();

private:
  /// Compute prefetch widths from dot encoding and shapes.
  /// Returns true if widths were set and this dot should be prefetched;
  /// false to skip this dot (e.g. kSize too small or dot not recognized).
  bool computePrefetchWidthForDotType(triton::DotOpInterface dot,
                                      Attribute dotEncoding,
                                      ArrayRef<int64_t> dShape, unsigned mSize,
                                      unsigned nSize, unsigned kSize,
                                      unsigned kWidth, bool transA, bool transB);

  std::tuple<unsigned, unsigned, unsigned>
  computePrefetchWidthFromNumInsts(unsigned mSize, unsigned nSize, unsigned kSize,
                       bool transA, bool transB, ArrayRef<unsigned> instrShape,
                       ArrayRef<unsigned> warpsPerCta, unsigned numInsts);

  Value generateLocalLoad(Value v, unsigned opIdx, bool isPrologue,
                          Attribute dotEncoding, OpBuilder &builder,
                          Value asyncWaitToken = Value(),
                          std::optional<int64_t> offsetM = std::nullopt,
                          std::optional<int64_t> shapeM = std::nullopt,
                          std::optional<int64_t> offsetN = std::nullopt,
                          std::optional<int64_t> shapeN = std::nullopt,
                          std::optional<int64_t> offsetK = std::nullopt,
                          std::optional<int64_t> shapeK = std::nullopt);

  Operation *generateDotsAndNonPrefetchingLocalLoads(Operation *dotOp,
                                                     Attribute dotEncoding,
                                                     OpBuilder &builder,
                                                     IRMapping &mapping,
                                                     scf::ForOp newForOp);

  void generatePrefetchingLocalLoads(Operation *dotOp, OpBuilder &builder,
                                     IRMapping &mapping,
                                     SmallVector<Value> &yieldValues,
                                     Value nextASource, Value nextAToken,
                                     Value nextBSource, Value nextBToken);

  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  /// dots to be prefetched (DotOp or DotScaledOp, both implement
  /// DotOpInterface)
  SetVector<Operation *> dots;
  /// Per-dot shared-memory sources, async-wait tokens, and intermediate
  /// elementwise op chains. Populated by initialize().
  ttg::DotPrefetchSources sources;
  /// Iter-arg indices for any non-loop-carried sources/tokens that we
  /// promote into new iter_args of the rewritten loop.
  ttg::DotPrefetchCarriedArgs carriedArgs;
  /// Original dot operand -> prologue-prefetched value.
  DenseMap<Value, Value> operand2headPrefetch;
  /// Cache of materializations created in the prologue, keyed by the original
  /// loop-body value. Lets us dedupe when multiple prefetches share the same
  /// expression DAG (e.g. an async_wait feeding both A and B).
  DenseMap<Value, Value> initMaterializations;

  /// Prefetch tile dimensions; set by computePrefetchWidths().
  unsigned prefetchWidthM;
  unsigned prefetchWidthN;
  unsigned prefetchWidthK;
  /// Store original kWidth to maintain when creating new local_loads.
  unsigned kWidth;
  /// Override for the target matrix-instructions-per-slice (numInsts). When
  /// nonzero it replaces the arch/dtype default; 0 means use the default.
  /// Plumbed from the `num-insts` pass option (mainly for testing).
  unsigned numInstsOverride;
};

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  SmallVector<Operation *> dotsInFor;
  for (Operation &op : *loop) {
    if (auto dotScaled = dyn_cast<triton::DotScaledOp>(&op)) {
      // TODO: need to support prefetching scales and slicing intermediate ops
      // before supporting DotScaledOp.
      if (dotScaled.getAScale() || dotScaled.getBScale()) {
        LDBG("DotScaledOp with scales is not supported.");
        LDBG(dotScaled);
        return failure();
      }
    }
    if (auto dotInterface = dyn_cast<triton::DotOpInterface>(&op)) {
      dotsInFor.push_back(&op);
    }
  }
  if (dotsInFor.empty()) {
    LDBG("No dots found in the loop.");
    return failure();
  }

  // TODO: enabling multiple dots per loop requires logic for prefetching
  // one dot's local_load inside another dot.
  if (dotsInFor.size() > 1) {
    LDBG("Multiple dots found in the loop; not yet supported.");
    return failure();
  }

  for (Operation *dot : dotsInFor) {
    auto dotInterface = cast<triton::DotOpInterface>(dot);
    auto aOpd = dotInterface.getA();
    auto bOpd = dotInterface.getB();
    auto aType = cast<RankedTensorType>(aOpd.getType());
    auto bType = cast<RankedTensorType>(bOpd.getType());
    auto dType = cast<RankedTensorType>(dot->getResult(0).getType());
    // The slicing logic below assumes 2-D operands. Skip batched dots.
    if (aType.getRank() != 2 || bType.getRank() != 2) {
      LDBG("Skipping dot with non-2D operands: " << *dot);
      continue;
    }
    auto aEnc = mlir::cast<ttg::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc = mlir::cast<ttg::DotOperandEncodingAttr>(bType.getEncoding());
    assert(aEnc.getKWidth() == bEnc.getKWidth());
    kWidth = aEnc.getKWidth();

    auto transOp = [&](Operation *op, int opdIdx) -> bool {
      while (op) {
        if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(op)) {
          auto srcType = localLoad.getSrc().getType();
          auto order = getOrder(srcType);
          return (order[0] == opdIdx);
        }
        if (op->getNumOperands() < 1)
          break;
        op = op->getOperand(0).getDefiningOp();
      }
      // For any other op type, fallback cautiously such that
      // subslices will be larger.
      return true;
    };

    // Get sizes for all three dimensions
    unsigned mSize = aType.getShape()[0];     // M dimension from operand A
    unsigned nSize = bType.getShape().back(); // N dimension from operand B
    unsigned kSize = aType.getShape().back(); // K dimension
    LDBG("dotShape: " << mSize << "x" << nSize << "x" << kSize);

    bool transA = transOp(aOpd.getDefiningOp(), 0);
    bool transB = transOp(bOpd.getDefiningOp(), 1);
    Attribute dotEncoding =
        cast<RankedTensorType>(dot->getResult(0).getType()).getEncoding();
    // TODO: calculating the prefetch(slicing) width also needs to examine
    // how the intermediate ops (between dot and local_load) are sliceable.
    if (!computePrefetchWidthForDotType(dotInterface, dotEncoding,
                                        dType.getShape(), mSize, nSize, kSize,
                                        kWidth, transA, transB)) {
      LDBG("computePrefetchWidthForDotType failed for dot: " << *dot);
      continue;
    }
    LDBG("subtile: " << prefetchWidthM << "x" << prefetchWidthN << "x"
                     << prefetchWidthK);
    assert(prefetchWidthM > 0 && prefetchWidthN > 0 && prefetchWidthK > 0);
    auto aVals = ttg::findLocalLoadForDotOperand(dotInterface.getA());
    auto bVals = ttg::findLocalLoadForDotOperand(dotInterface.getB());

    if (failed(aVals) || failed(bVals)) {
      LDBG("findLocalLoadForDotOperand failed for dot: " << *dot);
      continue;
    }
    Value aSmem = aVals->front();
    Value bSmem = bVals->front();
    // Skip dots whose operand is broadcast across CTA blocks; prefetching
    // would duplicate a broadcast and is semantically incorrect.
    if (ttg::isBroadcastedAlongCTABlock(aSmem) ||
        ttg::isBroadcastedAlongCTABlock(bSmem)) {
      LDBG("Skipping dot with broadcast operand: " << *dot);
      continue;
    }
    Value aToken = ttg::getLocalLoadToken(dotInterface.getA());
    Value bToken = ttg::getLocalLoadToken(dotInterface.getB());
    // Two patterns are accepted:
    //   (a) Loop-carried: both smem sources are iter_args. The original
    //       prefetch pattern; the prologue reads the init values and the
    //       end-of-iter prefetch reads yielded values.
    //   (b) Promotable: sources/tokens are in-body expressions (memdesc_index
    //       on a ring buffer plus optional async_wait) whose operands are
    //       themselves promotable. Prologue/yield materialize the expression
    //       with the induction var / iter_args remapped to init / yield.
    bool hasLoopCarriedSrc = ttg::isLoopCarriedValue(forOp, aSmem) &&
                             ttg::isLoopCarriedValue(forOp, bSmem);
    bool canPromoteSplitDot = (aToken || bToken) &&
                              ttg::isPromotableValue(forOp, aSmem) &&
                              ttg::isPromotableValue(forOp, bSmem) &&
                              ttg::isPromotableValue(forOp, aToken) &&
                              ttg::isPromotableValue(forOp, bToken);
    if (hasLoopCarriedSrc || canPromoteSplitDot) {
      dots.insert(dot);
      sources.aVals[dot] = *aVals;
      sources.bVals[dot] = *bVals;
      sources.aSource[dot] = aSmem;
      sources.bSource[dot] = bSmem;
      sources.aToken[dot] = aToken;
      sources.bToken[dot] = bToken;
    } else {
      LDBG("Dot operands are not prefetchable: " << *dot);
    }
  }
  if (dots.empty()) {
    LDBG("No prefetchable dots found in the loop.");
    return failure();
  }
  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (Operation *dot : dots) {
    auto dotInterface = cast<triton::DotOpInterface>(dot);
    Attribute dotEncoding =
        cast<RankedTensorType>(dot->getResult(0).getType()).getEncoding();
    // Materialize source and token expressions for the prologue, replacing
    // loop-carried iter_args with their init values and the induction var with
    // the lower bound. For loop-carried sources this simply returns the init
    // arg; for promoted expressions (e.g. memdesc_index / async_wait) this
    // clones the expression chain just before the loop.
    Value aSrcInit = ttg::materializeInitValue(forOp, sources.aSource[dot],
                                               builder, initMaterializations);
    Value aTokenInit = ttg::materializeInitValue(
        forOp, sources.aToken.lookup(dot), builder, initMaterializations);
    Value aPrefetched =
        generateLocalLoad(aSrcInit, 0, true, dotEncoding, builder, aTokenInit);
    ttg::clonePrefetchElementwiseOps(aPrefetched, sources.aVals[dot], builder);

    Value bSrcInit = ttg::materializeInitValue(forOp, sources.bSource[dot],
                                               builder, initMaterializations);
    Value bTokenInit = ttg::materializeInitValue(
        forOp, sources.bToken.lookup(dot), builder, initMaterializations);
    Value bPrefetched =
        generateLocalLoad(bSrcInit, 1, true, dotEncoding, builder, bTokenInit);
    ttg::clonePrefetchElementwiseOps(bPrefetched, sources.bVals[dot], builder);

    operand2headPrefetch[dotInterface.getA()] = aPrefetched;
    operand2headPrefetch[dotInterface.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  // Build the iter_arg list. First the existing init args, then any promoted
  // source/token iter_args required for in-body expressions, then the two
  // head-prefetch values (A and B) for each dot.
  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (Operation *dot : dots) {
    auto dotInterface = cast<triton::DotOpInterface>(dot);
    ttg::appendMaterializedLoopArgIfNeeded(
        forOp, dot, sources.aSource.lookup(dot), carriedArgs.aSource, loopArgs,
        builder, initMaterializations);
    ttg::appendMaterializedLoopArgIfNeeded(
        forOp, dot, sources.bSource.lookup(dot), carriedArgs.bSource, loopArgs,
        builder, initMaterializations);
    ttg::appendMaterializedLoopArgIfNeeded(
        forOp, dot, sources.aToken.lookup(dot), carriedArgs.aToken, loopArgs,
        builder, initMaterializations);
    ttg::appendMaterializedLoopArgIfNeeded(
        forOp, dot, sources.bToken.lookup(dot), carriedArgs.bToken, loopArgs,
        builder, initMaterializations);
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
    // For dots we synthesize a chain of sliced dots + interleaved local_loads
    // and don't need (or want) a verbatim clone of the original dot in the
    // new loop body. Skip the clone in that case to avoid leaving dead IR
    // behind (which would be DCE'd anyway, but emitting it is wasted work).
    Operation *newOp;
    if (dots.contains(&op)) {
      Attribute dotEncoding =
          cast<RankedTensorType>(op.getResult(0).getType()).getEncoding();
      newOp = generateDotsAndNonPrefetchingLocalLoads(&op, dotEncoding, builder,
                                                      mapping, newForOp);
    } else {
      newOp = builder.clone(op, mapping);
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // Build the yield. First the existing body yield operands (mapped through to
  // the new loop), then yielded values for any promoted source/token iter_args,
  // and finally the two next-iteration head prefetches per dot.
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (Operation *dot : dots) {
    // Compute next-iteration source/token once and reuse for both the
    // promoted iter-arg yield and the head-prefetch generation to avoid
    // cloning the same expression twice.
    Value nextASource =
        ttg::getNextTrackedValue(forOp, yieldOp, dot, /*isA=*/true,
                                 /*isToken=*/false, builder, mapping, sources);
    Value nextBSource =
        ttg::getNextTrackedValue(forOp, yieldOp, dot, /*isA=*/false,
                                 /*isToken=*/false, builder, mapping, sources);
    Value nextAToken =
        ttg::getNextTrackedValue(forOp, yieldOp, dot, /*isA=*/true,
                                 /*isToken=*/true, builder, mapping, sources);
    Value nextBToken =
        ttg::getNextTrackedValue(forOp, yieldOp, dot, /*isA=*/false,
                                 /*isToken=*/true, builder, mapping, sources);
    if (carriedArgs.contains(dot, /*isA=*/true, /*isToken=*/false))
      yieldValues.push_back(nextASource);
    if (carriedArgs.contains(dot, /*isA=*/false, /*isToken=*/false))
      yieldValues.push_back(nextBSource);
    if (carriedArgs.contains(dot, /*isA=*/true, /*isToken=*/true))
      yieldValues.push_back(nextAToken);
    if (carriedArgs.contains(dot, /*isA=*/false, /*isToken=*/true))
      yieldValues.push_back(nextBToken);
    generatePrefetchingLocalLoads(dot, builder, mapping, yieldValues,
                                  nextASource, nextAToken, nextBSource,
                                  nextBToken);
  }
  // Update ops of yield. After prefetching we always have at least the
  // head-prefetch values per dot, so yieldValues is never empty here.
  builder.setInsertionPointToEnd(newForOp.getBody());
  scf::YieldOp::create(builder, yieldOp.getLoc(), yieldValues);
  return newForOp;
}

//------------------------------------------------------------------------------
// computeDefaultNumInsts()
//------------------------------------------------------------------------------
//
// AMD-specific prefetch is based on prefetching enough data early enough to
// not be stalled by LDS latency, but not prefetch too much as this consumes
// excessive VGPRs and reduces cycles available for HBM-latency hiding.
//
// The degree of prefetching is expressed as `numInsts`, the target number of
// MFMA/WMMA instructions contained in one sub-tile (a single sliced dot). The
// dot is sliced into (totalWmmas / numWmmasPerTile) sub-tiles, so numInsts directly
// sets how many slices are produced. Since we only prefetch local_loads one sice
// in advance, we choose tiles whose cycles will take ~2X the lds latency; this ensures
// that even a local_load issued at the end of it's tile will be ready by the middle
// of the next tile. computeDefaultNumInsts() derives this from the GPU target and the
// matrix instruction's FMA count:
//
//   matrixCycles = ceil(numFMAsPerInst / fmasPerCycle)
//   numInsts     = pow2_ceil(2*ceil(LDS latency / matrixCycles))
//
//   * 16-bit MFMA (gfx94x/gfx95x): instrShape 16x16x16 = 4096 FMAs, ~256 FMAs/cyc
//     -> 16 cyc/MFMA; with ~48-cyc LDS latency, 2*ceil(48/16) = 6 -> pow2 = 8.
//   * fp8 MFMA: instrShape 16x16x32 = 8192 FMAs -> 32 cyc/MFMA; each instruction
//     packs more K, so fewer are needed: 2*ceil(48/32) = 4 -> 4.
//   * 16-bit WMMA (gfx1250): instrShape 16x16x32 = 8192 FMAs -> 32 cyc/WMMA; RDNA
//     sees ~2x the effective LDS latency (~96 cyc), 2*ceil(96/32) = 6 -> pow2 = 8.
//
// Because the instruction shape encodes the operand dtype (narrower operands have
// a larger K, hence more FMAs/instruction), the dtype's throughput effect falls
// out of numFMAsPerInst without a separate dtype branch.
//
// Larger LDS latency or fewer FMAs per instruction push numInsts up, which
// produces larger prefetchWidths, fewer slice/join ops, more VGPRs consumed,
// and more prefetched data in flight.
//
// computePrefetchWidthFromNumInsts() turns numInsts into the per-slice tile shape
// (prefetchWidthM, prefetchWidthN, prefetchWidthK) using these policies:
//
//   1. Arch minimum transpose width (`mtw`). When ds_read_tr is used to feed
//      an MFMA/WMMA operand (transA / transB), the corresponding non-K dim
//      and K must each be at least the arch's transpose width (32 / 64 / 128).
//      We bias `m`/`n`/`k` up to that floor for the transposed operand so we
//      don't slice below what ds_read_tr can produce in one shot.
//
//   2. Growth order. After accounting for the transpose-width floor we
//      iteratively double the slice tile dimensions until we've allocated
//      `numInsts` instructions to one slice. Growth order is M -> N -> K,
//      subject to not exceeding maxM / maxN / maxK (the number of instr-tiles
//      available along each dim) and not growing M when transA is set
//      (because M is already pinned by the transpose floor on operand A).
//
//   3. Row-vs-square preference. `preferSquare = false` lets M grow ahead of
//      N (`m <= n || !preferSquare` short-circuits to true), producing
//      row-shaped (M-heavy) tiles like 8x1 instead of square-ish 4x2 for
//      numInsts = 8. Empirically LLVM schedules row-shaped slices better;
//      flip preferSquare to true to experiment with square slices, which
//      changes (a) how many local_loads are needed to prefetch the first
//      `numInsts` matrix ops and (b) how local_loads interleave with them.
//
//   4. Power-of-2 tiles. The split/join helpers iteratively halve, so the
//      resulting tile dims must be powers of two. We return {0,0,0} from
//      computePrefetchWidthFromNumInsts() when this can't be satisfied, and skip
//      prefetching for that dot rather than asserting.
//
// computePrefetchWidthForDotType() also bails (returns false) when the dot's
// layout can't be described as a simple `warpsPerCTA` grid:
//
//   * MFMA with non-unit `tilesPerWarp` (warp does multiple contiguous tiles)
//   * WMMA with warp-swizzled ctaLayout (non-permutation linear layout)
//
// In both cases the slicing math below assumes the canonical
// `instrShape * warpsPerCTA` tiling and would mis-size slices. A more general
// LL-based analysis can be added later to lift these restrictions.
//
//------------------------------------------------------------------------------

// Compute the arch/dtype default for `numInsts` (the target number of matrix
// instructions worth of work per sliced dot) when the user does not override it
// (num-insts=0). The slice must be large enough to hide LDS read latency behind
// matrix-instruction issue: we want roughly 2x the LDS latency worth of compute
// in flight. The model is:
//
//   numFMAsPerInst = product(instrShape)              // e.g. MFMA 16x16x16 = 4096
//   cyclesPerInst  = numFMAsPerInst / numFMAsPerCycle // per-instruction issue cost
//   numInsts       = pow2_ceil(2 * (ldsLatency / cyclesPerInst))
//
// `numFMAsPerInst` is the number of fused multiply-adds a single matrix
// instruction performs (the product of the instruction shape dims).
// `numFMAsPerCycle` is the matrix unit's FMA throughput for the given arch/dtype,
// expressed as a representative instruction's FMA count divided by its issue
// cycles (e.g. gfx942 f16 16x16x16 over 16 cycles -> 256). Dividing the two
// yields the per-instruction issue cost, and the slice is sized at ~2x the LDS
// latency in instructions.
//
// `numInsts` is rounded up to the next power of two because computePrefetchWidthFromNumInsts
// grows the slice tile by doubling dimensions (consuming the budget via repeated
// halving), so only the power-of-two part actually translates into allocated
// matrix instructions.
//
// Selection is nested arch -> slower-operand dtype (the wider operand dictates
// matrix-unit throughput); each leaf picks the (ldsLatency, numFMAsPerCycle)
// approximations for that target/dtype.
//
// NOTE: the latency/throughput numbers below are best-effort approximations
// intended to be refined with profiling; the `num-insts` pass option /
// TRITON_HIP_LDS_PREFETCH_NUM_INSTS env var override them for tuning.
static unsigned computeDefaultNumInsts(StringRef arch,
                                       triton::DotOpInterface dot) {
  // (1) Operand element types; the slower (wider) operand dictates matrix-unit
  // throughput.
  auto aType = cast<RankedTensorType>(dot.getA().getType());
  auto bType = cast<RankedTensorType>(dot.getB().getType());
  unsigned aBits = aType.getElementType().getIntOrFloatBitWidth();
  unsigned bBits = bType.getElementType().getIntOrFloatBitWidth();
  unsigned slowerBits = std::max(aBits, bBits);

  // (2) Matrix instruction shape to determine how many cycles it takes;
  // e.g. 32x32x8 takes twice as long as 16x16x16.
  Attribute dEnc = cast<RankedTensorType>(dot->getResult(0).getType()).getEncoding();
  ArrayRef<unsigned> instrShape;
  if (auto mfmaEnc = dyn_cast<ttg::AMDMfmaEncodingAttr>(dEnc))
    instrShape = mfmaEnc.getInstrShape();
  else if (auto wmmaEnc = dyn_cast<ttg::AMDWmmaEncodingAttr>(dEnc))
    instrShape = wmmaEnc.getInstrShape();
  unsigned numFMAsPerInst = 1;
  for (unsigned d : instrShape)
    numFMAsPerInst *= d;

  // (3) Per-arch/dtype LDS latency and matrix-unit FMA throughput. The defaults
  // here are the fallback for unknown targets; each leaf below overrides them.
  unsigned numFMAsPerCycle = 16;
  unsigned ldsLatency = 64;
  if (arch == "gfx1250") {
    ldsLatency = 96;
    if (slowerBits <= 4) {
      numFMAsPerCycle = (32*16*128)/8; // 8192
    } else if (slowerBits <= 8) {
      numFMAsPerCycle = (16*16*128)/8; // 4096
    } else if (slowerBits <= 16) {
      numFMAsPerCycle =(16*16*32)/8; // 1024
    } else if (slowerBits <= 32) {
      numFMAsPerCycle =(16*16*4)/16; // 16
    }
  } else if (arch == "gfx950" || arch == "gfx951") { // mi350, mi355
    ldsLatency = 64;
    if (slowerBits <= 4) {
      numFMAsPerCycle = (32*32*32)/16; // 2048
    } else if (slowerBits <= 8) {
      numFMAsPerCycle = (32*32*16)/16; // 1024
    } else if (slowerBits <= 16) {
      numFMAsPerCycle =(32*32*8)/16; // 512
    } else if (slowerBits <= 32) {
      numFMAsPerCycle =(32*32*4)/64; // 64
    } else { // 64bit
      numFMAsPerCycle =(16*16*4)/64; // 16
    }
  } else if (arch == "gfx942") { // mi300
    ldsLatency = 64;
    if (slowerBits <= 8) {
      numFMAsPerCycle = (16*16*32)/16; // 512
    } else if (slowerBits <= 16) {
      numFMAsPerCycle =(16*16*16)/16; // 256
    } else if (slowerBits <= 32) {
      numFMAsPerCycle =(32*32*2)/32; // 64
    } else { // 64bit
      numFMAsPerCycle =(16*16*4)/32; // 32
    }
  }
  // Num cycles per WMMA instruction.
  unsigned cyclesPerInst = std::max(1u, numFMAsPerInst / numFMAsPerCycle);

  // Final calculation to ensure that each tile
  // will take ~2x the LDS latency to complete.
  unsigned numInsts = llvm::PowerOf2Ceil(2 * (ldsLatency / cyclesPerInst));
  LDBG("computeDefaultNumInsts(): lds=" << ldsLatency
                                         << ", numFMAsPerInst=" << numFMAsPerInst
                                         << ", numFMAsPerCycle=" << numFMAsPerCycle
                                         << ", cyclesPerInst=" << cyclesPerInst
                                         << " -> numInsts=" << numInsts);

  return numInsts;
}

// Top-level call to compute the prefetch width for a dot.
// It uses computeDefaultNumInsts() and computePrefetchWidthFromNumInsts()
// to compute the prefetch width.
bool Prefetcher::computePrefetchWidthForDotType(triton::DotOpInterface dot,
                                                Attribute dotEncoding,
                                                ArrayRef<int64_t> dShape,
                                                unsigned mSize, unsigned nSize,
                                                unsigned kSize, unsigned kWidth,
                                                bool transA, bool transB) {
  // Default `numInsts` is derived from the GPU target and the matrix
  // instruction's FMA count (see computeDefaultNumInsts). The `num-insts` pass
  // option overrides this when nonzero (mainly for testing).
  ModuleOp module = this->forOp.getOperation()->getParentOfType<ModuleOp>();
  std::optional<StringRef> arch = getAMDArch(module);
  StringRef archRef = arch ? *arch : StringRef();

  if (auto mfmaEnc = dyn_cast<ttg::AMDMfmaEncodingAttr>(dotEncoding)) {
    // The slice-size math below treats the dot as an
    // `instrShape * warpsPerCTA` grid, which is only valid for the default
    // tilesPerWarp = [1, 1] layout. Bail out for non-unit tilesPerWarp
    // (e.g. layouts produced to resolve LDS partition conflicts) until a
    // more general LL-based analysis is plumbed through.
    if (!mfmaEnc.hasUnitTilesPerWarp()) {
      LDBG("Skipping MFMA with non-unit tilesPerWarp: " << mfmaEnc);
      return false;
    }
    // Target a sliced tile spanning ~2x the LDS latency in MFMAs; see the
    // "Slicing policy" block above and computeDefaultNumInsts(). The
    // `num-insts` pass option overrides this default when nonzero.
    auto instrShape = mfmaEnc.getInstrShape();
    unsigned numInsts = numInstsOverride
                            ? numInstsOverride
                            : computeDefaultNumInsts(archRef, dot);
    std::tie(prefetchWidthM, prefetchWidthN, prefetchWidthK) =
        computePrefetchWidthFromNumInsts(mSize, nSize, kSize, transA, transB, instrShape,
                             mfmaEnc.getWarpsPerCTA(), numInsts);
  } else if (auto wmmaEnc = dyn_cast<ttg::AMDWmmaEncodingAttr>(dotEncoding)) {
    // WMMA stores its warp distribution as a general linear layout in
    // ctaLayout rather than a plain warpsPerCTA grid. Reduce to warpsPerCTA
    // via getWarpsPerCTA() only when ctaLayout is a permutation matrix (i.e.
    // a 1:1 warp-per-tile mapping without warp swizzling). Bail otherwise --
    // warpsPerCTA is not well-defined for warp-swizzled WMMA layouts (which
    // are used e.g. to resolve partition conflicts on gfx1250) and the
    // slicing computation below would silently produce the wrong tile size.
    if (!ttg::isPermutationMatrixLayout(wmmaEnc.getCtaLayout())) {
      LDBG("Skipping warp-swizzled WMMA layout: " << wmmaEnc);
      return false;
    }
    // Target a sliced tile spanning ~2x the LDS latency in WMMAs; see the
    // "Slicing policy" block above and computeDefaultNumInsts(). The
    // `num-insts` pass option overrides this default when nonzero.
    auto instrShape = wmmaEnc.getInstrShape();
    unsigned numInsts = numInstsOverride
                            ? numInstsOverride
                            : computeDefaultNumInsts(archRef, dot);
    auto warpsPerCTA = ttg::getWarpsPerCTA(wmmaEnc, dShape);
    std::tie(prefetchWidthM, prefetchWidthN, prefetchWidthK) =
        computePrefetchWidthFromNumInsts(mSize, nSize, kSize, transA, transB, instrShape,
                             warpsPerCTA, numInsts);
  } else {
    return false;
  }
  // computePrefetchWidthFromNumInsts returns {0, 0, 0} when the resulting tile is not a
  // power of 2; treat that as "skip this dot".
  return prefetchWidthM > 0 && prefetchWidthN > 0 && prefetchWidthK > 0;
}

// After determining the minimum number of matrix instructions per slice,
// this function computes the tile shape which is later used for slicing.
// May choose tiles larger than numInsts when slicing along a dimension
// may have performance consequences, e.g. too small a size to transpose
// at highest throughput.
std::tuple<unsigned, unsigned, unsigned> Prefetcher::computePrefetchWidthFromNumInsts(
    unsigned mSize, unsigned nSize, unsigned kSize, bool transA, bool transB,
    ArrayRef<unsigned> instrShape, ArrayRef<unsigned> warpsPerCta,
    unsigned numInsts) {

  // Arch minimum transpose width (mtw): when ds_read_tr feeds the matrix
  // instruction we must not slice an operand below this width on its
  // transposed dim. See policy block above.
  ModuleOp module = this->forOp.getOperation()->getParentOfType<ModuleOp>();
  std::optional<StringRef> arch = getAMDArch(module);
  if (!arch)
    return {mSize, nSize, kSize};
  std::string archStr = arch->str();
  unsigned mtw = 32;
  if (archStr == "gfx1250") {
    mtw = 128;
  } else if (archStr == "gfx942" || archStr == "gfx950" ||
             archStr == "gfx951") {
    mtw = 64;
  }

  LDBG("instrShape: " << instrShape[0] << "x" << instrShape[1] << "x"
                      << instrShape[2]);
  LDBG("warpsPerCta: " << warpsPerCta[0] << "x" << warpsPerCta[1]);
  // m, n, k are the per-slice counts of matrix instructions along each dim.
  // maxM/maxN/maxK are the totals available -- the slice is at most the
  // whole dot along each dim.
  unsigned m = 1, n = 1, k = 1;
  unsigned maxM = std::max<unsigned>(1, mSize / (instrShape[0] * warpsPerCta[0]));
  unsigned maxN = std::max<unsigned>(1, nSize / (instrShape[1] * warpsPerCta[1]));
  unsigned maxK = std::max<unsigned>(1, kSize / (instrShape[2]));
  LDBG("maxInsts: " << maxM << "x" << maxN << "x" << maxK << " (p=" << numInsts
                    << ")");
  // (1) Pin transposed operand dims to the arch transpose-width floor.
  if (transA) {
    m = std::max<unsigned>(m, std::min<unsigned>(maxM, mtw / instrShape[0]));
    k = std::max<unsigned>(k, std::min<unsigned>(maxK, mtw / instrShape[2]));
  }
  if (transB) {
    n = std::max<unsigned>(n, std::min<unsigned>(maxN, mtw / instrShape[1]));
    k = std::max<unsigned>(k, std::min<unsigned>(maxK, mtw / instrShape[2]));
  }
  LDBG("numInstrsPerSubtile: " << m << "x" << n << "x" << k
                               << " (based on Trans)");
  numInsts /= (m * n * k);
  // (2,3) Grow the slice to the remaining numInsts budget in M -> N -> K
  // order. preferSquare = false biases growth toward row-shaped tiles, which
  // LLVM currently schedules better than square tiles for this code pattern.
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
  LDBG("numInstrsPerSubtile: " << m << "x" << n << "x" << k);
  LDBG("numSubTiles: " << maxM / m << "x" << maxN / n << "x" << maxK / k);
  // convert num ops to CTA tile shape
  m *= instrShape[0] * warpsPerCta[0];
  n *= instrShape[1] * warpsPerCta[1];
  k *= instrShape[2];
  m = std::min<unsigned>(m, mSize);
  n = std::min<unsigned>(n, nSize);
  k = std::min<unsigned>(k, kSize);
  // The slice/join logic (which iteratively halves) requires power-of-2
  // tile sizes; signal failure to the caller by returning {0, 0, 0} so the
  // dot is skipped instead of asserting.
  if (!llvm::isPowerOf2_32(m) || !llvm::isPowerOf2_32(n) ||
      !llvm::isPowerOf2_32(k)) {
    LDBG("computePrefetchWidthFromNumInsts: non-power-of-2 tile (" << m << "x" << n << "x"
                                                       << k << "); skipping");
    return {0, 0, 0};
  }
  return {m, n, k};
}

// Generates a new local_load operation for a subtile based on the
// original local_load, v, and the offsets and shapes of the subtile.
// Since dots have 3D slicing, the MemDescSubslice for local loads
// will have 2D offsets and shapes.
Value Prefetcher::generateLocalLoad(
    Value v, unsigned opIdx, bool isPrologue, Attribute dotEncoding,
    OpBuilder &builder, Value asyncWaitToken, std::optional<int64_t> offsetM,
    std::optional<int64_t> shapeM, std::optional<int64_t> offsetN,
    std::optional<int64_t> shapeN, std::optional<int64_t> offsetK,
    std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<ttg::MemDescType>(v.getType());
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

  Value newSmem = ttg::MemDescSubsliceOp::create(
      builder, v.getLoc(),
      ttg::MemDescType::get(shape, elementType, type.getEncoding(),
                            type.getMemorySpace(), type.getMutableMemory(),
                            type.getAllocShape()),
      v, offset);
  auto dotOperandEnc = ttg::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding, kWidth);
  Value prefetchSlice = ttg::LocalLoadOp::create(
      builder, v.getLoc(),
      RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem,
      asyncWaitToken);
  return prefetchSlice;
}

// Insert a sched.barrier for MFMA/WMMA and lds ops.
// We pin everything except mfma/wmma so the backend can't pull loads/ALU
// across the barrier; `all_vmem` already covers both reads and writes so we
// don't need to also OR in `vmem_read`/`vmem_write`.
// Even with sched barriers, we still rely on llvm scheduler to
// (1) scheduler the ds_loads relatively early in the tile and
// (2) interleave the ds_loads and wmmas.
void insertSchedBarrier(OpBuilder &builder, Location loc) {
  int32_t mask = (int32_t)mlir::amdgpu::sched_barrier_opt_enum::valu |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::salu |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::all_vmem |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::vmem_read |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::vmem_write |
                 (int32_t)mlir::amdgpu::sched_barrier_opt_enum::transcendental;
  ROCDL::SchedBarrier::create(builder, loc, mask);
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
  SmallVector<RankedTensorType> typesBeforeSplitting;
  SmallVector<Value> mSlices = splitValueAlongAxis(
      cOperand, numSlicesM, mAxis, typesBeforeSplitting, loc, builder);
  // Slice c opds along N
  int32_t numSlicesN = totalN / prefetchWidthN;
  for (int32_t mIdx = 0; mIdx < numSlicesM; ++mIdx) {
    int32_t mOff = mIdx * prefetchWidthM;
    SmallVector<Value> mnSlices = splitValueAlongAxis(
        mSlices[mIdx], numSlicesN, nAxis, typesBeforeSplitting, loc, builder);
    for (int32_t nIdx = 0; nIdx < numSlicesN; ++nIdx) {
      int32_t nOff = nIdx * prefetchWidthN;
      mnToDot[{mOff, nOff}] = mnSlices[nIdx];
    }
  }

  // The current-iteration source/token for A and B are independent of the
  // (k, m, n) tile coordinate, so compute them once up front. They are looked
  // up through carriedArgs so that both loop-carried iter_args and in-body
  // promoted expressions work uniformly.
  Value aSrc = ttg::getCurrentTrackedValue(forOp, dotOp, /*isA=*/true,
                                           /*isToken=*/false, newForOp, mapping,
                                           sources, carriedArgs);
  Value aTok = ttg::getCurrentTrackedValue(forOp, dotOp, /*isA=*/true,
                                           /*isToken=*/true, newForOp, mapping,
                                           sources, carriedArgs);
  Value bSrc = ttg::getCurrentTrackedValue(forOp, dotOp, /*isA=*/false,
                                           /*isToken=*/false, newForOp, mapping,
                                           sources, carriedArgs);
  Value bTok = ttg::getCurrentTrackedValue(forOp, dotOp, /*isA=*/false,
                                           /*isToken=*/true, newForOp, mapping,
                                           sources, carriedArgs);

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
            aSlice = generateLocalLoad(aSrc, 0, false, dotEncoding, builder,
                                       aTok, mOff, prefetchWidthM, std::nullopt,
                                       std::nullopt, kOff, prefetchWidthK);
            ttg::clonePrefetchElementwiseOps(aSlice, sources.aVals[dotOp],
                                             builder);
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
            bSlice = generateLocalLoad(bSrc, 1, false, dotEncoding, builder,
                                       bTok, std::nullopt, std::nullopt, nOff,
                                       prefetchWidthN, kOff, prefetchWidthK);
            ttg::clonePrefetchElementwiseOps(bSlice, sources.bVals[dotOp],
                                             builder);
            bSlices[nOff] = bSlice;
          } else {
            // Reuse the opd previously created during mOff=0 for mOff>0.
            bSlice = bSlices[nOff];
          }
        }

        // Sched.barrier after local_loads and before dot to enforce
        // prefetching; 0th dot skipped b/c prefetched.
        if (mOff > 0 || nOff > 0 || kOff > 0)
          insertSchedBarrier(builder, loc);

        if (lastDotOp)
          builder.setInsertionPointAfter(lastDotOp);
        Value cSlice = mnToDot[{mOff, nOff}];
        auto dType = cast<RankedTensorType>(cSlice.getType());
        Operation *newDot = createDotOp(dotOp, builder, loc, dType, aSlice,
                                        bSlice, cSlice, &mapping);
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
    Value mJoin = joinValuesAlongAxis(mnSlices, nAxis, typesBeforeSplitting,
                                      loc, builder);
    mJoins.push_back(mJoin);
  }
  // Join d opds along M
  Value result =
      joinValuesAlongAxis(mJoins, mAxis, typesBeforeSplitting, loc, builder);
  Operation *newOp = result.getDefiningOp();
  // Reset insertion point to before the last dot for the prefetched local loads
  builder.setInsertionPoint(lastDotOp);
  return newOp;
}

// Generates the prefetched local loads which are for dot[m=0,n=0,k=0] of the
// next iteration. These are issued at the end of the current iteration and
// yielded so they become the head-prefetch values on entry to the next
// iteration.
void Prefetcher::generatePrefetchingLocalLoads(
    Operation *dotOp, OpBuilder &builder, IRMapping &mapping,
    SmallVector<Value> &yieldValues, Value nextASource, Value nextAToken,
    Value nextBSource, Value nextBToken) {
  Attribute dotEncoding =
      cast<RankedTensorType>(dotOp->getResult(0).getType()).getEncoding();
  Value aToYield =
      generateLocalLoad(nextASource, 0, true, dotEncoding, builder, nextAToken);
  ttg::clonePrefetchElementwiseOps(aToYield, sources.aVals[dotOp], builder);
  yieldValues.push_back(aToYield);

  Value bToYield =
      generateLocalLoad(nextBSource, 1, true, dotEncoding, builder, nextBToken);
  ttg::clonePrefetchElementwiseOps(bToYield, sources.bVals[dotOp], builder);
  yieldValues.push_back(bToYield);
  // Sched.barrier after local_loads (and before dot) to enforce prefetching.
  insertSchedBarrier(builder, dotOp->getLoc());
}

bool hasPrecedingCondBarrier(scf::ForOp forOp) {
  for (Operation *op = forOp->getPrevNode(); op; op = op->getPrevNode())
    if (isa<triton::amdgpu::CondBarrierOp>(op))
      return true;
  return false;
}

} // anonymous namespace
} // namespace amdgpu
} // namespace triton

struct TritonAMDGPULdsPrefetchPass
    : public impl::TritonAMDGPULdsPrefetchBase<TritonAMDGPULdsPrefetchPass> {
  using Base::Base;

  void runOnOperation() override {
    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    ttg::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                      &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp forOp) {
      // Don't run LdsPrefetch on loops preceded by a conditional barrier,
      // as this signifies that PingPong scheduling succeeded.
      if (triton::amdgpu::hasPrecedingCondBarrier(forOp)) {
        LDBG("Skipping loop (CondBarrierOp signifies PingPong).");
        return;
      }

      triton::amdgpu::Prefetcher prefetcher(forOp, numInstsOverride);

      if (prefetcher.initialize().failed()) {
        LDBG("LdsPrefetching failed.");
        return;
      }

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      // replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
      LDBG("LdsPrefetching succeeded.");
    });
  }
};

} // namespace mlir
