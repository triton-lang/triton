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
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
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
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  // TODO: add a hook to infer prefetchWidth
  unsigned prefetchWidth = 32;

  /// dots to be prefetched
  SetVector<triton::DotOp> dots;
  DenseMap<Value, Value> dot2aSource;
  DenseMap<Value, Value> dot2bSource;
  DenseMap<Value, Value> dot2aToken;
  DenseMap<Value, Value> dot2bToken;
  DenseMap<Value, SmallVector<Value>> dot2aVals;
  DenseMap<Value, SmallVector<Value>> dot2bVals;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;
  DenseMap<Value, Value> initMaterializations;

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         Value token = Value(),
                         std::optional<int64_t> offsetK = std::nullopt,
                         std::optional<int64_t> shapeK = std::nullopt);

  bool isLoopCarriedValue(Value v);
  Value getIncomingValue(Value v);
  Value getYieldValue(Value v);
  bool isPromotableValue(Value v);
  Value cloneLoopValue(Value v, OpBuilder &builder,
                       llvm::function_ref<Value(BlockArgument)> mapBlockArg,
                       DenseMap<Value, Value> &cache);
  Value materializeInitValue(Value v, OpBuilder &builder,
                             DenseMap<Value, Value> &cache);
  Value materializeYieldValue(Value v, OpBuilder &builder,
                              const IRMapping &mapping,
                              DenseMap<Value, Value> &cache);

  void cloneElementwiseOps(Value &bRem, const SmallVector<Value> &vals,
                           OpBuilder &builder);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
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
                                   Value token,
                                   std::optional<int64_t> offsetK,
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

bool Prefetcher::isLoopCarriedValue(Value v) {
  auto arg = dyn_cast_if_present<BlockArgument>(v);
  return arg && arg.getOwner()->getParentOp() == forOp.getOperation();
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
  // Loop-carried block arguments can be remapped to either the init or yield
  // value when materializing the prologue/epilogue.
  if (isLoopCarriedValue(v))
    return true;
  Operation *op = v.getDefiningOp();
  // Other block arguments / values without a defining op are assumed safe.
  if (!op)
    return true;
  // Values defined outside this loop body are already available in the
  // preheader, so we do not need to clone loop-local IR for them.
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
  return llvm::all_of(op->getOperands(),
                      [this](Value operand) { return isPromotableValue(operand); });
}

Value Prefetcher::cloneLoopValue(
    Value v, OpBuilder &builder,
    llvm::function_ref<Value(BlockArgument)> mapBlockArg,
    DenseMap<Value, Value> &cache) {
  if (!v)
    return Value();
  if (auto it = cache.find(v); it != cache.end())
    return it->second;
  if (auto arg = dyn_cast<BlockArgument>(v)) {
    Value mapped = mapBlockArg(arg);
    cache[v] = mapped;
    return mapped;
  }
  Operation *op = v.getDefiningOp();
  if (!op || op->getBlock() != forOp.getBody()) {
    cache[v] = v;
    return v;
  }

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
        if (arg.getOwner()->getParentOp() != forOp.getOperation())
          return arg;
        return forOp.getTiedLoopInit(arg)->get();
      },
      cache);
}

Value Prefetcher::materializeYieldValue(Value v, OpBuilder &builder,
                                        const IRMapping &mapping,
                                        DenseMap<Value, Value> &cache) {
  return cloneLoopValue(
      v, builder,
      [this, &mapping](BlockArgument arg) -> Value {
        if (arg.getOwner()->getParentOp() != forOp.getOperation())
          return arg;
        return mapping.lookupOrDefault(getYieldValue(arg));
      },
      cache);
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  auto getEncoding = [](Value v) {
    return cast<TensorOrMemDesc>(v.getType()).getEncoding();
  };

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
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
        // NYI for other encodings, for example if we have transpose
        // in the chain
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
    auto aEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc =
        mlir::cast<triton::gpu::DotOperandEncodingAttr>(bType.getEncoding());
    int aKWidth = aEnc.getKWidth();
    int bKWidth = bEnc.getKWidth();
    assert(aKWidth == bKWidth);

    auto kSize = aType.getShape().back();

    // works better with nvidia tensor cores
    unsigned elementWidth = aType.getElementTypeBitWidth();
    if (aKWidth == 0)
      prefetchWidth = 256 / elementWidth;
    else
      prefetchWidth = 8 * aKWidth;

    // Skip prefetching if kSize is less than prefetchWidth
    if (kSize < prefetchWidth)
      continue;
    auto aVals = getPrefetchSrc(dot.getA());
    auto bVals = getPrefetchSrc(dot.getB());

    if (aVals.size() && bVals.size()) {
      Value aSmem = aVals.front();
      Value bSmem = bVals.front();
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
    Value aPrefetched =
        generatePrefetch(materializeInitValue(dot2aSource[dot], builder,
                                             initMaterializations),
                         0, true, dotEncoding, builder,
                         materializeInitValue(dot2aToken.lookup(dot), builder,
                                              initMaterializations));
    cloneElementwiseOps(aPrefetched, dot2aVals[dot], builder);
    Value bPrefetched =
        generatePrefetch(materializeInitValue(dot2bSource[dot], builder,
                                             initMaterializations),
                         1, true, dotEncoding, builder,
                         materializeInitValue(dot2bToken.lookup(dot), builder,
                                              initMaterializations));
    cloneElementwiseOps(bPrefetched, dot2bVals[dot], builder);

    operand2headPrefetch[dot.getA()] = aPrefetched;
    operand2headPrefetch[dot.getB()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getInitArgs())
    loopArgs.push_back(v);
  DenseMap<Operation *, unsigned> dot2aCarriedTokenArg;
  DenseMap<Operation *, unsigned> dot2bCarriedTokenArg;
  for (triton::DotOp dot : dots) {
    if (Value aToken = dot2aToken.lookup(dot); aToken && !isLoopCarriedValue(aToken)) {
      dot2aCarriedTokenArg[dot] = loopArgs.size();
      loopArgs.push_back(
          materializeInitValue(aToken, builder, initMaterializations));
    }
    if (Value bToken = dot2bToken.lookup(dot); bToken && !isLoopCarriedValue(bToken)) {
      dot2bCarriedTokenArg[dot] = loopArgs.size();
      loopArgs.push_back(
          materializeInitValue(bToken, builder, initMaterializations));
    }
    loopArgs.push_back(operand2headPrefetch[dot.getA()]);
    loopArgs.push_back(operand2headPrefetch[dot.getB()]);
  }

  auto newForOp =
      scf::ForOp::create(builder, forOp.getLoc(), forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  IRMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

  auto getMappedValue = [&mapping](Value v) -> Value {
    return v ? mapping.lookupOrDefault(v) : Value();
  };
  auto getCurrentSource = [this, &getMappedValue](triton::DotOp dot,
                                           bool isA) -> Value {
    Value source = isA ? dot2aSource.lookup(dot) : dot2bSource.lookup(dot);
    return getMappedValue(source);
  };
  auto getCurrentToken = [this, &mapping, &newForOp, &dot2aCarriedTokenArg,
                          &dot2bCarriedTokenArg](triton::DotOp dot,
                                                 bool isA) -> Value {
    Value token = isA ? dot2aToken.lookup(dot) : dot2bToken.lookup(dot);
    if (!token)
      return Value();
    if (isLoopCarriedValue(token))
      return mapping.lookupOrDefault(token);
    auto &argMap = isA ? dot2aCarriedTokenArg : dot2bCarriedTokenArg;
    auto it = argMap.find(dot);
    if (it == argMap.end())
      return Value();
    return newForOp.getRegionIterArgs()[it->second];
  };
  auto getNextSource = [this, &builder, &mapping](triton::DotOp dot,
                                                  bool isA) -> Value {
    Value source = isA ? dot2aSource.lookup(dot) : dot2bSource.lookup(dot);
    if (!source)
      return Value();
    if (isLoopCarriedValue(source))
      return mapping.lookupOrDefault(getYieldValue(source));
    DenseMap<Value, Value> yieldCache;
    return materializeYieldValue(source, builder, mapping, yieldCache);
  };
  auto getNextToken = [this, &mapping](triton::DotOp dot, bool isA) -> Value {
    Value token = isA ? dot2aToken.lookup(dot) : dot2bToken.lookup(dot);
    if (!token)
      return Value();
    if (isLoopCarriedValue(token))
      return mapping.lookupOrDefault(getYieldValue(token));
    return mapping.lookupOrDefault(token);
  };

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
      // prefetched dot
      Operation *firstDot = builder.clone(*dot, mapping);
      if (Value a = operand2headPrefetch.lookup(dot.getA()))
        firstDot->setOperand(
            0, newForOp.getTiedLoopRegionIterArg(&*a.use_begin()));
      if (Value b = operand2headPrefetch.lookup(dot.getB()))
        firstDot->setOperand(
            1, newForOp.getTiedLoopRegionIterArg(&*b.use_begin()));

      // remaining part
      int64_t kOff = prefetchWidth;
      int64_t kRem = dot.getA().getType().getShape().back() - prefetchWidth;
      Operation *prevDot = firstDot;
      if (kRem == 0) {
        // There is only one dot while prefetchWidth == kSize so delay issuing
        // it. Meanwhile, newOp should be set to firstDot to make sure the dot
        // result is updated to yield.
        builder.setInsertionPoint(prevDot);
        newOp = firstDot;
      }

      while (kRem != 0) {
        // int64_t kShape = largestPow2(kRem);
        int64_t kShape = prefetchWidth;
        auto insertionPoint = builder.saveInsertionPoint();
        builder.setInsertionPoint(prevDot);
        Value aRem = generatePrefetch(getCurrentSource(dot, true), 0, false,
                                      dotEncoding, builder,
                                      getCurrentToken(dot, true), kOff, kShape);
        cloneElementwiseOps(aRem, dot2aVals[dot], builder);
        Value bRem = generatePrefetch(getCurrentSource(dot, false), 1, false,
                                      dotEncoding, builder,
                                      getCurrentToken(dot, false), kOff, kShape);
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
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookupOrDefault(v));
  for (triton::DotOp dot : dots) {
    if (dot2aCarriedTokenArg.contains(dot))
      yieldValues.push_back(getNextToken(dot, true));
    if (dot2bCarriedTokenArg.contains(dot))
      yieldValues.push_back(getNextToken(dot, false));
    Attribute dotEncoding = dot.getType().getEncoding();
    Value aToYield = generatePrefetch(getNextSource(dot, true), 0, true,
                                      dotEncoding, builder,
                                      getNextToken(dot, true));
    cloneElementwiseOps(aToYield, dot2aVals[dot], builder);
    yieldValues.push_back(aToYield);
    // bToYield
    Value bToYield = generatePrefetch(getNextSource(dot, false), 1, true,
                                      dotEncoding, builder,
                                      getNextToken(dot, false));
    cloneElementwiseOps(bToYield, dot2bVals[dot], builder);
    yieldValues.push_back(bToYield);
  }
  // Update ops of yield
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
    triton::gpu::ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns,
                                                              &getContext());
    if (mlir::applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
            .failed()) {
      signalPassFailure();
    }
    getOperation()->walk([&](scf::ForOp forOp) {
      Prefetcher prefetcher(forOp);

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
