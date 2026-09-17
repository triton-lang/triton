//===----------------------------------------------------------------------===//
//
// This pass rewrites selected tt.dot loops to pull the prefetched head out of
// the loop and to prefetch the next iteration's operands at the end of the
// loop. The concrete shape is covered by split_pipelined_mmav2_loads in
// test/TritonGPU/prefetch.mlir.
//
// Example:
// %loop = scf.for ... {
//   %wait = ttg.async_wait %tok0, %tok1 {num = 4 : i32}
//   %a_view = ttg.memdesc_index %a[%idx_next]
//   %a_val = ttg.local_load %a_view token %wait
//   %b_view = ttg.memdesc_index %b[%idx_next]
//   %b_val = ttg.local_load %b_view token %wait
//   %acc_next = tt.dot %a_val, %b_val, %acc
//   scf.yield %idx_next, %acc_next
// }
//
// becomes:
// %a_view0 = ttg.memdesc_index %a[%idx_next0]
// %b_view0 = ttg.memdesc_index %b[%idx_next0]
// %wait0 = ttg.async_wait %tok0, %tok1 {num = 4 : i32}
// %a0 = ttg.local_load %a_view0 token %wait0
// %b0 = ttg.local_load %b_view0 token %wait0
// %loop = scf.for ... iter_args(..., %wait = %wait0, %a_prefetch = %a0,
//                               %b_prefetch = %b0) {
//   %wait_next = ttg.async_wait %tok0, %tok1 {num = 4 : i32}
//   %a_rem = ttg.local_load %a_tail token %wait
//   %b_rem = ttg.local_load %b_tail token %wait
//   %dot0 = tt.dot %a_prefetch, %b_prefetch, %acc
//   %a_next = ttg.local_load %next_a_head token %wait_next
//   %b_next = ttg.local_load %next_b_head token %wait_next
//   %acc_next = tt.dot %a_rem, %b_rem, %dot0
//   scf.yield ..., %wait_next, %a_next, %b_next
// }
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-prefetch"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUPREFETCH
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

class Prefetcher {
  struct CarriedArgs {
    DenseMap<Operation *, unsigned> aSource;
    DenseMap<Operation *, unsigned> bSource;
    DenseMap<Operation *, unsigned> a;
    DenseMap<Operation *, unsigned> b;
  };

  /// Loop being rewritten.
  scf::ForOp forOp;
  /// Original loop terminator, used to recover yielded values.
  scf::YieldOp yieldOp;
  unsigned prefetchWidth = 32;
  int computeCapability;

  /// Dots that will be rewritten to use prologue/next-iteration prefetches.
  SetVector<triton::DotOp> dots;
  DenseMap<Value, Value> dot2aSource;
  DenseMap<Value, Value> dot2bSource;
  DenseMap<Value, Value> dot2aToken;
  DenseMap<Value, Value> dot2bToken;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  /// Original dot operand -> prologue-prefetched value.
  DenseMap<Value, Value> operand2headPrefetch;
  DenseMap<Value, Value> initMaterializations;

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         Value token = Value(),
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);
  unsigned getKWidthScale(Attribute dotEncoding, Type elementType) const;
  unsigned getDotOperandKWidth(Attribute dotEncoding, Type elementType) const;
  unsigned getPrefetchWidth(Attribute dotEncoding, Type elementType,
                            unsigned kWidth) const;

  bool isLoopCarriedValue(Value v);
  Value getIncomingValue(Value v);
  Value getYieldValue(Value v);
  bool isPromotableValue(Value v);
  Value cloneLoopValue(Value v, OpBuilder &builder,
                       llvm::function_ref<Value(BlockArgument)> mapBlockArg,
                       DenseMap<Value, Value> &cache);
  Value materializeInitValue(Value v, OpBuilder &builder,
                             DenseMap<Value, Value> &cache);
  void appendMaterializedLoopArgIfNeeded(
      triton::DotOp dot, Value value, DenseMap<Operation *, unsigned> &argMap,
      SmallVector<Value> &loopArgs, OpBuilder &builder);
  Value getTrackedValue(triton::DotOp dot, bool isA, bool isToken);
  const DenseMap<Operation *, unsigned> &
  getCarriedArgMap(const CarriedArgs &carriedArgs, bool isA, bool isToken);
  Value getCurrentTrackedValue(triton::DotOp dot, bool isA, bool isToken,
                               scf::ForOp newForOp, IRMapping &mapping,
                               const CarriedArgs &carriedArgs);
  Value getNextTrackedValue(triton::DotOp dot, bool isA, bool isToken,
                            OpBuilder &builder, IRMapping &mapping);
  SmallVector<Value> createLoopArgs(OpBuilder &builder,
                                    CarriedArgs &carriedArgs);
  void cloneLoopBody(scf::ForOp newForOp, OpBuilder &builder,
                     IRMapping &mapping, const CarriedArgs &carriedArgs);
  SmallVector<Value> createYieldValues(OpBuilder &builder, IRMapping &mapping,
                                       const CarriedArgs &carriedArgs);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp, int computeCapability)
      : forOp(forOp), computeCapability(computeCapability) {
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

Value Prefetcher::generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   Value token, std::optional<int64_t> offsetK,
                                   std::optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = cast<triton::gpu::MemDescType>(v.getType());
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  auto rank = shape.size();
  SmallVector<int32_t> offset(rank, 0);
  Type elementType = type.getElementType();

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? rank - 1 : rank - 2;

  offset[kIdx] = isPrologue ? 0 : prefetchWidth;
  shape[kIdx] = isPrologue ? prefetchWidth : (shape[kIdx] - prefetchWidth);

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
      builder.getContext(), opIdx, dotEncoding, prefetchWidth / 8);
  Value prefetchSlice = triton::gpu::LocalLoadOp::create(
      builder, v.getLoc(),
      RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem, token);

  return prefetchSlice;
}

unsigned Prefetcher::getPrefetchWidth(Attribute dotEncoding, Type elementType,
                                      unsigned kWidth) const {
  if (kWidth == 0)
    return 256 / elementType.getIntOrFloatBitWidth();
  return 8 * kWidth;
}

bool Prefetcher::isLoopCarriedValue(Value v) {
  auto arg = dyn_cast_if_present<BlockArgument>(v);
  return arg && arg.getOwner() == forOp.getBody() &&
         arg.getArgNumber() >= forOp.getNumInductionVars();
}

Value Prefetcher::getIncomingValue(Value v) {
  if (!isLoopCarriedValue(v))
    return Value();
  auto arg = cast<BlockArgument>(v);
  return forOp.getTiedLoopInit(arg)->get();
}

Value Prefetcher::getYieldValue(Value v) {
  if (!isLoopCarriedValue(v))
    return Value();
  auto arg = cast<BlockArgument>(v);
  unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
  return yieldOp.getOperand(yieldIdx);
}

bool Prefetcher::isPromotableValue(Value v) {
  // Null operands are treated as trivially promotable.
  // e.g., local_load no tokens
  if (!v)
    return true;
  if (auto arg = dyn_cast<BlockArgument>(v))
    return arg.getOwner() != forOp.getBody() || isLoopCarriedValue(arg) ||
           arg == forOp.getInductionVar();
  // Loop-carried block arguments can be remapped to either the init value or
  // the yielded next-iteration value during rewrite.
  if (isLoopCarriedValue(v))
    return true;
  Operation *op = v.getDefiningOp();
  // Other block arguments / values without a defining op are assumed safe.
  if (!op)
    return true;
  // Values defined outside this loop body are already available where we
  // materialize the prologue/yield expressions, so they do not need cloning.
  if (op->getBlock() != forOp.getBody())
    return true;
  // Nested control flow is not handled by the cloning logic below.
  if (op->getNumRegions() != 0)
    return false;
  // Only clone simple elementwise/constant ops plus the specific loop-local
  // ops needed to rebuild the async-wait + memdesc-index chain.
  if (!op->hasTrait<OpTrait::Elementwise>() &&
      !op->hasTrait<OpTrait::ConstantLike>() &&
      !isa<triton::gpu::AsyncWaitOp, triton::gpu::MemDescIndexOp>(op))
    return false;
  // Every operand must also be promotable, otherwise the whole expression is
  // rejected.
  return llvm::all_of(op->getOperands(), [this](Value operand) {
    return isPromotableValue(operand);
  });
}

Value Prefetcher::cloneLoopValue(
    Value v, OpBuilder &builder,
    llvm::function_ref<Value(BlockArgument)> mapBlockArg,
    DenseMap<Value, Value> &cache) {
  // Null values are allowed for optional operands such as local_load tokens.
  if (!v)
    return Value();
  // Reuse previously cloned values when reconstructing a shared expression DAG.
  if (auto it = cache.find(v); it != cache.end())
    return it->second;
  // Block arguments are remapped by the caller depending on whether we are
  // materializing the loop init or the yielded next-iteration value.
  if (auto arg = dyn_cast<BlockArgument>(v))
    return cache[v] = mapBlockArg(arg);
  Operation *op = v.getDefiningOp();
  // Values defined outside this loop body can be reused directly.
  if (op->getBlock() != forOp.getBody())
    return cache[v] = v;

  // Recursively rebuild the loop-local expression with remapped operands.
  IRMapping operandMapping;
  for (Value operand : op->getOperands())
    operandMapping.map(operand,
                       cloneLoopValue(operand, builder, mapBlockArg, cache));
  Operation *clonedOp = builder.clone(*op, operandMapping);
  for (auto [result, clonedResult] :
       llvm::zip(op->getResults(), clonedOp->getResults()))
    cache[result] = clonedResult;
  return cache[v];
}

Value Prefetcher::materializeInitValue(Value v, OpBuilder &builder,
                                       DenseMap<Value, Value> &cache) {
  return cloneLoopValue(
      v, builder,
      [this](BlockArgument arg) -> Value {
        if (arg.getOwner() != forOp.getBody())
          return arg;
        if (arg == forOp.getInductionVar())
          return forOp.getLowerBound();
        return forOp.getTiedLoopInit(arg)->get();
      },
      cache);
}

void Prefetcher::appendMaterializedLoopArgIfNeeded(
    triton::DotOp dot, Value value, DenseMap<Operation *, unsigned> &argMap,
    SmallVector<Value> &loopArgs, OpBuilder &builder) {
  if (!value || isLoopCarriedValue(value))
    return;
  argMap[dot] = loopArgs.size();
  loopArgs.push_back(
      materializeInitValue(value, builder, initMaterializations));
}

Value Prefetcher::getTrackedValue(triton::DotOp dot, bool isA, bool isToken) {
  if (isToken)
    return isA ? dot2aToken.lookup(dot) : dot2bToken.lookup(dot);
  return isA ? dot2aSource.lookup(dot) : dot2bSource.lookup(dot);
}

const DenseMap<Operation *, unsigned> &
Prefetcher::getCarriedArgMap(const CarriedArgs &carriedArgs, bool isA,
                             bool isToken) {
  if (isToken)
    return isA ? carriedArgs.a : carriedArgs.b;
  return isA ? carriedArgs.aSource : carriedArgs.bSource;
}

Value Prefetcher::getCurrentTrackedValue(triton::DotOp dot, bool isA,
                                         bool isToken, scf::ForOp newForOp,
                                         IRMapping &mapping,
                                         const CarriedArgs &carriedArgs) {
  Value value = getTrackedValue(dot, isA, isToken);
  if (!value)
    return Value();
  // If token or source value is initially loop carried. It means local_load is
  // done outside of the loop and we can directly use the tracked value
  if (isLoopCarriedValue(value))
    return mapping.lookupOrDefault(value);
  const auto &argMap = getCarriedArgMap(carriedArgs, isA, isToken);
  auto it = argMap.find(dot);
  if (it == argMap.end())
    // The arg is invalid for prefetching
    return isToken ? Value() : mapping.lookupOrDefault(value);
  // The arg is initalized outside of the loop and passed into the new loop as
  // an argument
  return newForOp.getRegionIterArgs()[it->second];
}

Value Prefetcher::getNextTrackedValue(triton::DotOp dot, bool isA, bool isToken,
                                      OpBuilder &builder, IRMapping &mapping) {
  Value value = getTrackedValue(dot, isA, isToken);
  if (!value)
    return Value();
  if (isLoopCarriedValue(value))
    return mapping.lookupOrDefault(getYieldValue(value));

  DenseMap<Value, Value> yieldCache;
  return cloneLoopValue(
      value, builder,
      [this, &builder, &mapping](BlockArgument arg) -> Value {
        if (arg.getOwner() != forOp.getBody())
          return arg;
        if (arg == forOp.getInductionVar())
          return arith::AddIOp::create(builder, forOp.getLoc(),
                                       mapping.lookupOrDefault(arg),
                                       forOp.getStep());
        return mapping.lookupOrDefault(getYieldValue(arg));
      },
      yieldCache);
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();
  auto kBlock = StringAttr::get(forOp.getContext(), "block");

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };
  auto isBroadcasted = [kBlock, &getEncoding](Value v) {
    auto cgaLayout = getCGALayout(getEncoding(v)).getLinearLayout();
    if (!cgaLayout.hasInDim(kBlock))
      return false;
    return cgaLayout.getFreeVariableMasks()[kBlock] != 0;
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop) {
    if (auto dotOp = dyn_cast<triton::DotOp>(op)) {
      // Only accepts dotOps encoded as Nvidia MMA v2 or AMD MFMA
      auto dstMmaEnc =
          dyn_cast<NvidiaMmaEncodingAttr>(getEncoding(dotOp.getResult()));
      auto dstMfmaEnc =
          dyn_cast<AMDMfmaEncodingAttr>(getEncoding(dotOp.getResult()));
      if (!dstMfmaEnc && (!dstMmaEnc || dstMmaEnc.getVersionMajor() != 2))
        // Don't rewrite if any other type is found.
        return failure();
      dotsInFor.push_back(dotOp);
    }
    if (isa<triton::nvidia_gpu::TMAOpInterface>(op)) {
      // Don't rewrite if syncTMACopy or asyncTMACopy is found since they may
      // have dependencies with the dot op that are not handled by the current
      // implementation.
      return failure();
    }
  }

  if (dotsInFor.empty())
    return failure();

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  // Walk back from the dot operand to the shared-memory value consumed by the
  // local_load chain.
  auto getPrefetchSrc = [](Value v) -> SmallVector<Value> {
    // Walk backwards through the single-use chain until we find local_load.
    Operation *op = v.getDefiningOp();
    if (!op)
      return {};
    bool foundConvertFromShared = false;
    SmallVector<Value> rets;
    rets.push_back(op->getResult(0));
    LDBG("Prefetch src: " << *op);
    while (op) {
      if (!op->getResult(0).hasOneUse())
        break;
      if (auto load = dyn_cast<triton::gpu::LocalLoadOp>(op)) {
        rets.push_back(load.getSrc());
        // Only handle the direct dot-operand load chain for now.
        if (isa<DotOperandEncodingAttr>(load.getType().getEncoding()))
          foundConvertFromShared = true;
        break;
      }
      if (op->getNumOperands() != 1)
        break;
      rets.push_back(op->getOperand(0));
      op = op->getOperand(0).getDefiningOp();
      if (op)
        LDBG("op: " << *op);
    }
    std::reverse(rets.begin(), rets.end());

    if (foundConvertFromShared)
      return rets;
    return {};
  };

  auto getLoadToken = [](Value v) -> Value {
    Operation *op = v.getDefiningOp();
    while (op) {
      if (auto load = dyn_cast<triton::gpu::LocalLoadOp>(op))
        return load.getToken();
      if (op->getNumOperands() != 1)
        break;
      op = op->getOperand(0).getDefiningOp();
    }
    return Value();
  };

  for (triton::DotOp dot : dotsInFor) {
    auto aType = dot.getA().getType();
    auto bType = dot.getB().getType();
    auto dotEncoding = dot.getType().getEncoding();
    auto aEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(bType.getEncoding());
    int aKWidth = aEnc.getKWidth();
    int bKWidth = bEnc.getKWidth();
    assert(aKWidth == bKWidth);

    auto kSize = aType.getShape().back();

    // Match the chunk width expected by the dot operand encoding.
    prefetchWidth =
        getPrefetchWidth(dotEncoding, aType.getElementType(), aKWidth);

    // Skip prefetching if kSize is less than prefetchWidth
    if (kSize < prefetchWidth)
      continue;
    auto aVals = getPrefetchSrc(dot.getA());
    auto bVals = getPrefetchSrc(dot.getB());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
      if (isBroadcasted(aSmem) || isBroadcasted(bSmem))
        continue;
      dot2aVals[dot] = aVals;
      dot2bVals[dot] = bVals;
      dot2aSource[dot] = aSmem;
      dot2bSource[dot] = bSmem;
      dot2aToken[dot] = getLoadToken(dot.getA());
      dot2bToken[dot] = getLoadToken(dot.getB());
      Value aHeaderDef = getIncomingValue(aSmem);
      Value bHeaderDef = getIncomingValue(bSmem);
      bool hasLoopCarriedSrc = aHeaderDef && bHeaderDef;
      bool canPromoteSplitDot =
          (dot2aToken[dot] || dot2bToken[dot]) && isPromotableValue(aSmem) &&
          isPromotableValue(bSmem) && isPromotableValue(dot2aToken[dot]) &&
          isPromotableValue(dot2bToken[dot]);
      if (hasLoopCarriedSrc || canPromoteSplitDot) {
        dots.insert(dot);
      }
    }
  }

  if (dots.empty())
    return failure();
  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (triton::DotOp dot : dots) {
    Attribute dotEncoding = dot.getType().getEncoding();
    Value aPrefetched = generatePrefetch(
        materializeInitValue(dot2aSource[dot], builder, initMaterializations),
        0, true, dotEncoding, builder,
        materializeInitValue(dot2aToken.lookup(dot), builder,
                             initMaterializations));
    cloneElementwiseOps(aPrefetched, dot2aVals[dot], builder);
    Value bPrefetched = generatePrefetch(
        materializeInitValue(dot2bSource[dot], builder, initMaterializations),
        1, true, dotEncoding, builder,
        materializeInitValue(dot2bToken.lookup(dot), builder,
                             initMaterializations));
    cloneElementwiseOps(bPrefetched, dot2bVals[dot], builder);

    operand2headPrefetch[dot.getA()] = aPrefetched;
    operand2headPrefetch[dot.getB()] = bPrefetched;
  }
}

SmallVector<Value> Prefetcher::createLoopArgs(OpBuilder &builder,
                                              CarriedArgs &carriedArgs) {
  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  for (triton::DotOp dot : dots) {
    appendMaterializedLoopArgIfNeeded(dot, dot2aSource.lookup(dot),
                                      carriedArgs.aSource, loopArgs, builder);
    appendMaterializedLoopArgIfNeeded(dot, dot2bSource.lookup(dot),
                                      carriedArgs.bSource, loopArgs, builder);
    appendMaterializedLoopArgIfNeeded(dot, dot2aToken.lookup(dot),
                                      carriedArgs.a, loopArgs, builder);
    appendMaterializedLoopArgIfNeeded(dot, dot2bToken.lookup(dot),
                                      carriedArgs.b, loopArgs, builder);
    loopArgs.push_back(operand2headPrefetch[dot.getA()]);
    loopArgs.push_back(operand2headPrefetch[dot.getB()]);
  }
  return loopArgs;
}

void Prefetcher::cloneLoopBody(scf::ForOp newForOp, OpBuilder &builder,
                               IRMapping &mapping,
                               const CarriedArgs &carriedArgs) {
  // Keep late-sunk ops before the loop terminator.
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
        auto dot = dyn_cast<triton::DotOp>(def);
        if (dot && dots.contains(dot)) {
          setInsertionPointBeforeYield(builder, newForOp);
        }
      }
    }
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<triton::DotOp>(&op);
    if (dot && dots.contains(dot)) {
      Attribute dotEncoding = dot.getType().getEncoding();
      // First dot uses the values prefetched before entering the loop.
      Operation *firstDot = builder.clone(*dot, mapping);
      if (Value a = operand2headPrefetch.lookup(dot.getA()))
        firstDot->setOperand(
            0, newForOp.getTiedLoopRegionIterArg(&*a.use_begin()));
      if (Value b = operand2headPrefetch.lookup(dot.getB()))
        firstDot->setOperand(
            1, newForOp.getTiedLoopRegionIterArg(&*b.use_begin()));

      // Emit additional dots for the remainder of K after the prefetched head.
      const int64_t kChunk = prefetchWidth;
      int64_t kOff = kChunk;
      int64_t kRem = dot.getA().getType().getShape().back() - kChunk;
      Operation *prevDot = firstDot;
      if (kRem == 0) {
        // There is only one dot while prefetchWidth == kSize so delay issuing
        // it. Meanwhile, newOp should be set to firstDot to make sure the dot
        // result is updated to yield.
        builder.setInsertionPoint(prevDot);
        newOp = firstDot;
      }

      while (kRem != 0) {
        int64_t kShape = kChunk;
        auto insertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPoint(prevDot);
        Value aRem = generatePrefetch(
            getCurrentTrackedValue(dot, /*isA=*/true, /*isToken=*/false,
                                   newForOp, mapping, carriedArgs),
            0, false, dotEncoding, builder,
            getCurrentTrackedValue(dot, /*isA=*/true, /*isToken=*/true,
                                   newForOp, mapping, carriedArgs),
            kOff, kShape);
        cloneElementwiseOps(aRem, dot2aVals[dot], builder);
        Value bRem = generatePrefetch(
            getCurrentTrackedValue(dot, /*isA=*/false, /*isToken=*/false,
                                   newForOp, mapping, carriedArgs),
            1, false, dotEncoding, builder,
            getCurrentTrackedValue(dot, /*isA=*/false, /*isToken=*/true,
                                   newForOp, mapping, carriedArgs),
            kOff, kShape);
        cloneElementwiseOps(bRem, dot2bVals[dot], builder);
        builder.restoreInsertionPoint(insertionPoint);
        newOp = builder.clone(*dot, mapping);
        newOp->setOperand(0, aRem);
        newOp->setOperand(1, bRem);
        newOp->setOperand(2, prevDot->getResult(0));
        prevDot = newOp;
        kOff += kShape;
        kRem -= kShape;
        if (kRem == 0) {
          // We want to delay issuing the last dot as long as possible, ideally
          // until after the prefetch.  To accomplish this, set the insertion
          // point above the dot.  If we find anything dependent on the dot (at
          // the top of this loop), we resume inserting after it.
          builder.setInsertionPoint(prevDot);
        }
      }
    }
    // Forward all uses in the cloned body to the rewritten operations.
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }
}

SmallVector<Value>
Prefetcher::createYieldValues(OpBuilder &builder, IRMapping &mapping,
                              const CarriedArgs &carriedArgs) {
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (triton::DotOp dot : dots) {
    Value nextASource = getNextTrackedValue(
        dot, /*isA=*/true, /*isToken=*/false, builder, mapping);
    Value nextBSource = getNextTrackedValue(
        dot, /*isA=*/false, /*isToken=*/false, builder, mapping);
    Value nextAToken = getNextTrackedValue(dot, /*isA=*/true, /*isToken=*/true,
                                           builder, mapping);
    Value nextBToken = getNextTrackedValue(dot, /*isA=*/false, /*isToken=*/true,
                                           builder, mapping);

    if (carriedArgs.aSource.contains(dot))
      yieldValues.push_back(nextASource);
    if (carriedArgs.bSource.contains(dot))
      yieldValues.push_back(nextBSource);
    if (carriedArgs.a.contains(dot))
      yieldValues.push_back(nextAToken);
    if (carriedArgs.b.contains(dot))
      yieldValues.push_back(nextBToken);
    Attribute dotEncoding = dot.getType().getEncoding();
    Value aToYield = generatePrefetch(nextASource, 0, true, dotEncoding,
                                      builder, nextAToken);
    cloneElementwiseOps(aToYield, dot2aVals[dot], builder);
    yieldValues.push_back(aToYield);
    Value bToYield = generatePrefetch(nextBSource, 1, true, dotEncoding,
                                      builder, nextBToken);
    cloneElementwiseOps(bToYield, dot2bVals[dot], builder);
    yieldValues.push_back(bToYield);
  }
  return yieldValues;
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);
  CarriedArgs carriedArgs;
  SmallVector<Value> loopArgs = createLoopArgs(builder, carriedArgs);

  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  cloneLoopBody(newForOp, builder, mapping, carriedArgs);

  SmallVector<Value> yieldValues =
      createYieldValues(builder, mapping, carriedArgs);
  // Replace the loop terminator with the rebuilt yield.
  builder.setInsertionPointToEnd(newForOp.getBody());
  if (!yieldValues.empty())
    scf::YieldOp::create(builder, yieldOp.getLoc(), yieldValues);
  return newForOp;
}

} // anonymous namespace

struct PrefetchPass : public impl::TritonGPUPrefetchBase<PrefetchPass> {
  void runOnOperation() override {

    // Canonicalize convert ops to make the pattern matching easier.
    RewritePatternSet cleanUpPatterns(&getContext());
    ModuleOp m = getOperation();
    int computeCapability = 0;
    if (auto targetAttr =
            m->getAttrOfType<StringAttr>(triton::gpu::AttrTargetName);
        targetAttr && targetAttr.getValue().starts_with("cuda:")) {
      computeCapability = getNVIDIAComputeCapability(m);
    }
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    m->walk([&](scf::ForOp forOp) {
      Prefetcher prefetcher(forOp, computeCapability);

      if (prefetcher.initialize().failed())
        return;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

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
