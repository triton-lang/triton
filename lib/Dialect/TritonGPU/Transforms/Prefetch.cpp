#include "mlir/IR/BlockAndValueMapping.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

using namespace mlir;

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  // TODO: add a hook to infer prefetchWidth
  unsigned prefetchWidth = 16;

  /// dots to be prefetched
  SetVector<Value> dots;
  /// dot => dot operand
  DenseMap<Value, Value> dot2aArg;
  DenseMap<Value, Value> dot2aDef;
  DenseMap<Value, Value> dot2bArg;
  DenseMap<Value, Value> dot2bDef;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  /// operand => defining
  DenseMap<Value, Value> operand2headDef;

  LogicalResult isForOpOperand(Value v);

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrefetch,
                         Attribute mmaEncoding, OpBuilder &builder);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

Value Prefetcher::generatePrefetch(Value v, unsigned opIdx, bool isPrefetch,
                                   Attribute mmaEncoding, OpBuilder &builder) {
  // opIdx: 0 => a, 1 => b
  auto type = v.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  SmallVector<int64_t> offset{0, 0};
  Type elementType = type.getElementType();

  auto intAttr = [&](int64_t val) { return builder.getI64IntegerAttr(val); };

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? 1 : 0;
  offset[kIdx] = isPrefetch ? 0 : prefetchWidth;
  shape[kIdx] = isPrefetch ? prefetchWidth : (shape[kIdx] - prefetchWidth);

  Value newSmem = builder.create<tensor::ExtractSliceOp>(
      v.getLoc(),
      // TODO: encoding?
      RankedTensorType::get(shape, elementType, type.getEncoding()), v,
      SmallVector<OpFoldResult>{intAttr(offset[0]), intAttr(offset[1])},
      SmallVector<OpFoldResult>{intAttr(shape[0]), intAttr(shape[1])},
      SmallVector<OpFoldResult>{intAttr(1), intAttr(1)});

  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, mmaEncoding);
  Value prefetchSlice = builder.create<triton::gpu::ConvertLayoutOp>(
      v.getLoc(), RankedTensorType::get(shape, elementType, dotOperandEnc),
      newSmem);

  return prefetchSlice;
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  SmallVector<triton::DotOp> dotsInFor;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op))
      dotsInFor.push_back(dotOp);

  if (dotsInFor.empty())
    return failure();

  auto getIncomingOp = [this](Value v) -> Value {
    if (auto arg = v.dyn_cast<BlockArgument>())
      if (arg.getOwner()->getParentOp() == forOp.getOperation())
        return forOp.getOpOperandForRegionIterArg(arg).get();
    return Value();
  };

  auto getYieldOp = [this](Value v) -> Value {
    auto arg = v.cast<BlockArgument>();
    unsigned yieldIdx = arg.getArgNumber() - forOp.getNumInductionVars();
    return yieldOp.getOperand(yieldIdx);
  };

  // Only prefetch loop arg
  for (triton::DotOp dot : dotsInFor) {
    if (Value op = getIncomingOp(dot.a())) {
      dot2aDef[dot] = op;
      dot2aArg[dot] = dot.a();
      dot2aYield[dot] = getYieldOp(dot.a());
      dots.insert(dot);
    }
    if (Value op = getIncomingOp(dot.b())) {
      dot2bDef[dot] = op;
      dot2bArg[dot] = dot.b();
      dot2bYield[dot] = getYieldOp(dot.b());
      dots.insert(dot);
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (Value dot : dots) {
    auto mmaEncoding = dot.getType()
                           .cast<RankedTensorType>()
                           .getEncoding()
                           .cast<triton::gpu::MmaEncodingAttr>();
    if (Value aDef = dot2aDef.lookup(dot)) {
      Value newA =
          generatePrefetch(aDef, 0, /*isPrefetch*/ true, mmaEncoding, builder);
      operand2headDef[dot.getDefiningOp<triton::DotOp>().a()] = newA;
    }
    if (Value bDef = dot2bDef.lookup(dot)) {
      Value newB =
          generatePrefetch(bDef, 1, /*isPrefetch*/ true, mmaEncoding, builder);
      operand2headDef[dot.getDefiningOp<triton::DotOp>().b()] = newB;
    }
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getIterOperands())
    loopArgs.push_back(v);
  for (Value dot : dots) {
    if (Value a = dot2aArg.lookup(dot))
      loopArgs.push_back(operand2headDef[a]);
    if (Value b = dot2bArg.lookup(dot))
      loopArgs.push_back(operand2headDef[b]);
  }

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  builder.setInsertionPointToStart(newForOp.getBody());
  BlockAndValueMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);

  for (Operation &op : forOp.getBody()->without_terminator()) {
    Operation *newOp = nullptr;
    if (auto dot = dyn_cast<triton::DotOp>(&op)) {
      auto mmaEncoding = dot.getType()
                             .cast<RankedTensorType>()
                             .getEncoding()
                             .cast<triton::gpu::MmaEncodingAttr>();
      // prefetched dot
      Operation *firstDot = builder.clone(*dot, mapping);
      if (Value a = operand2headDef.lookup(dot.a()))
        firstDot->setOperand(
            0, newForOp.getRegionIterArgForOpOperand(*a.use_begin()));
      if (Value b = operand2headDef.lookup(dot.b()))
        firstDot->setOperand(
            1, newForOp.getRegionIterArgForOpOperand(*b.use_begin()));

      // remaining part (Note it's possible that dot.a() is not in mapping)
      Value aRem = mapping.lookupOrNull(dot.a());
      Value bRem = mapping.lookupOrNull(dot.b());
      if (Value a = dot2aArg.lookup(dot))
        aRem =
            generatePrefetch(mapping.lookup(a), 0, false, mmaEncoding, builder);
      if (Value b = dot2bArg.lookup(dot))
        bRem =
            generatePrefetch(mapping.lookup(b), 1, false, mmaEncoding, builder);
      newOp = builder.clone(*dot, mapping);
      // Use sliced a & b
      if (aRem && aRem != mapping.lookup(dot.a()))
        newOp->setOperand(0, aRem);
      if (bRem && bRem != mapping.lookup(dot.b()))
        newOp->setOperand(1, bRem);
      newOp->setOperand(2, firstDot->getResult(0));
    } else {
      newOp = builder.clone(op, mapping);
    }
    // update mapping of results
    for (unsigned dstIdx : llvm::seq(unsigned(0), op.getNumResults()))
      mapping.map(op.getResult(dstIdx), newOp->getResult(dstIdx));
  }

  // prefetch next iteration
  SmallVector<Value> yieldValues;
  for (Value v : forOp.getBody()->getTerminator()->getOperands())
    yieldValues.push_back(mapping.lookup(v));
  for (Value dot : dots) {
    auto mmaEncoding = dot.getType()
                           .cast<RankedTensorType>()
                           .getEncoding()
                           .cast<triton::gpu::MmaEncodingAttr>();
    if (Value a = dot2aYield.lookup(dot))
      yieldValues.push_back(
          generatePrefetch(mapping.lookup(a), 0, true, mmaEncoding, builder));
    if (Value b = dot2bYield.lookup(dot))
      yieldValues.push_back(
          generatePrefetch(mapping.lookup(b), 1, true, mmaEncoding, builder));
  }
  // Update ops of yield
  builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
}

struct PrefetchPass : public TritonGPUPrefetchBase<PrefetchPass> {
  void runOnOperation() override {
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

    // // TODO: Can we use canonicalizer?
    // // a & b in `dot a, b, c` should be of DotOperand layout
    // getOperand->walk([&](triton::DotOp dotOp) {
    //   //
    // });
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPrefetchPass() {
  return std::make_unique<PrefetchPass>();
}