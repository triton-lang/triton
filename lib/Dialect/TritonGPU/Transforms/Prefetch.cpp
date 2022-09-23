#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace {

class Prefetcher {
  /// cache the ForOp we are working on
  scf::ForOp forOp;
  /// cache the YieldOp of this ForOp
  scf::YieldOp yieldOp;
  ///
  unsigned prefetchWidth;

  /// dots to be prefetched
  SetVector<Value> dots;
  /// dot => dot operand
  MapVector<Value, Value> dot2a;
  MapVector<Value, Value> dot2b;
  /// operand => defining
  DenseMap<Value, Value> operand2headDef;
  DenseMap<Value, Value> operand2LoopArg;

  LogicalResult isForOpOperand(Value v);

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrefetch,
                         Attribute mmaEncoding, OpBuilder &builder);

public:
  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();
};

Value Prefetcher::generatePrefetch(Value v, unsigned opIdx, bool isPrefetch,
                                   Attribute mmaEncoding, OpBuilder &builder) {
  // opIdx: 0 => a, 1 => b
  auto type = v.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape = type.getShape();
  SmallVector<int64_t> offset{0, 0};
  Type elementType = type.getElementType();

  auto intAttr = [&](int64_t val) { return builder.getI64IntegerAttr(val); }

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? 1 : 0;
  offset[kIdx] = isPrefetch? 0 : prefetchWidth;
  shape[kIdx] = isPrefetch? prefetchWidth : (k - prefetchWidth);


  Value newSmem = builder.create<tensor::ExtractSliceOp>(
    v.getLoc(),
    // TODO: encoding?
    RankedTensorType::get(shape, elementType, type.getEncoding()),
    SmallVector<OpFoldResult>{intAttr(offset[0]), intAttr(offset[1])},
    SmallVector<OpFoldResult>{intAttr(shape[0]), intAttr(shape[1])},
    SmallVector<OpFoldResult>{intAttr(1), intAttr(1)}
  );

  auto dotOperandEnc = 
    triton::gpu::DotOperandEncodingAttr::get(builder.getContext(), opIdx,
                                              mmaEncoding);
  Value prefetchSlice = builder.create<triton::gpu::ConvertLayoutOp>(
    v.getLoc(),
    RankedTensorType::get(shape, elementType, dotOperandEnc), newSmem
  );

  return prefetchSlice;
}

LogicalResult Prefetcher::initialize() {
  Block *loop = forOp.getBody();

  SmallVector<triton::DotOp> allDots;
  for (Operation &op : *loop)
    if (auto dotOp = dyn_cast<triton::DotOp>(op))
      allDots.push_back(dotOp);

  if (allDots.empty())
    return failure();

  auto getForOpArg = [&forOp](Value v) -> Value {
    if (v.hasOneUse() && v.use_begin()->getOwner() == forOp.getOperation()) {
      return forOp.getRegionIterArgForOpOperand(*v.use_begin());
    }
    return Value();
  }

  // Only prefetch loop arg
  for (triton::DotOp dot : allDots) {
    if (Value arg = getForOpArg(dot.a())) {
      dot2a[dot] = dot.a();
      operand2LoopArg[dot.a()] = arg;
      dots.push_back(dot);
    }
    if (Value arg = getForOpArg(dot.b())) {
      dot2b[dot] = dot.b();
      operand2LoopArg[dot.b()] = arg;
      dots.push_back(dot);
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (Value dot : dots) {
    auto mmaEncoding = dot.getType().cast<RankedTensorType>().getEncoding()
                          .cast<triton::gpu::MmaEncodingAttr>();
    if (Value a = dot2a.lookup(dot)) {
      Value newA = generatePrefetch(a, 0, /*isPrefetch*/true, mmaEncoding,
                                    builder);
      operand2headDef[a] = newA;
    }
    if (Value b = dot2b.lookup(dot)) {
      Value newB = generatePrefetch(b, 1, /*isPrefetch*/true, mmaEncoding,
                                    builder);
      operand2headDef[b] = newB;
    }
  }
}

void Prefetcher::createnewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getIterOperands())
    loopArgs.push_back(v);
  for (Value dot : dots) {
    if (Value a = dot2a.lookup(dot))
      loopArgs.push_back(operand2headDef[a]);
    if (Value b = dot2b.lookup(dot))
      loopArgs.push_back(operand2headDef[b]);
  }
  
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto dot = dyn_cast<triton::DotOp>(&op)) {
      Value firstDot = builder.clone(dot, ...);
      Value aRem = dot.a();
      Value bRem = dot.b();
      if (Value a = dot2a.lookup(dot))
        aRem = generatePrefetch(a, 0, false, mmaEncoding, builder);
      if (Value b = dot2b.lookup(dot))
        bRem = generatePrefetch(b, 1, false, mmaEncoding, builder);
      Value restDot = builder.clone(dot, ...);
      dot.replaceAllUsesWith(restDot);
      continue;
    }
    builder.clone(op);
  }

  // prefetch next itartion
  SmallVector<Value> yieldValues;
  for (Value dot : dots) {
    auto mmaEncoding = dot.getType().cast<RankedTensorType>().getEncoding()
                          .cast<triton::gpu::MmaEncodingAttr>();
    if (Value a = dot2a.lookup(dot))
      yieldValues.push_back(generatePrefetch(a, 0, true, mmaEncoding, builder));
    if (Value b = dot2b.lookup(dot))
      yieldValues.push_back(generatePrefetch(b, 1, true, mmaEncoding, builder));
  }
  // Update ops of yield
  builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
}

struct PrefetchPass : public TritonGPUPrefetchBase<PrefetchPass> {
  void runOnOperation() override {
    getOperation->walk([&](scr::ForOp forOp) {
      //

      if (prefetcher.initialize().failed())
        return;

      prefetcher.emitPrologue();

      scf::ForOp newForOp = prefetcher.createNewForOp();

      // replace the original loop
      for (unsigned i = 0; i < forOp->getNumResults(); ++i)
        forOp->getResult(i).replaceAllUsesWith(newForOp->getResult(i));
      forOp->erase();
    });

    // a & b in `dot a, b, c` should be of DotOperand layout
    getOperand->walk([&](triton::DotOp dotOp) {
      //
    });
  }
};

} // anonymous namespace


std::unique_ptr<Pass> mlir::createTritonGPUPrefetchPass() {
  return std::make_unique<PrefetchPass>();
}