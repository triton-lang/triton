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
// %a_tmp = tensor.extract_slice %a[0, 0] [128, 16]
// %a_prefetch = triton_gpu.convert_layout %a_tmp
// scf.for %iv = ... iter_args(%a_buf = %a, ..., %a_prefetch_arg = %a_prefetch)
// {
//   %x = tt.dot %a_arg, %b, %c
//   %a_tmp_rem = tensor.extract_slice %a_buf[0, 16] [128, 16]
//   %a_prefetch_next = triton_gpu.convert_layout %a_tmp_rem
//   ...
//   scf.yield %next_a, ..., %a_prefetch_next
// }
//===----------------------------------------------------------------------===//

#include "mlir/IR/BlockAndValueMapping.h"
#include "triton/Analysis/Utility.h"
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
  DenseMap<Value, Value> dot2aLoopArg;
  DenseMap<Value, Value> dot2aHeaderDef;
  DenseMap<Value, Value> dot2bLoopArg;
  DenseMap<Value, Value> dot2bHeaderDef;
  DenseMap<Value, Value> dot2aYield;
  DenseMap<Value, Value> dot2bYield;
  /// operand => defining
  DenseMap<Value, Value> operand2headPrefetch;

  LogicalResult isForOpOperand(Value v);

  Value generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                         Attribute dotEncoding, OpBuilder &builder,
                         llvm::Optional<int64_t> offsetK = llvm::None,
                         llvm::Optional<int64_t> shapeK = llvm::None);

public:
  Prefetcher() = delete;

  Prefetcher(scf::ForOp forOp) : forOp(forOp) {
    yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  }

  LogicalResult initialize();

  void emitPrologue();

  scf::ForOp createNewForOp();
};

Value Prefetcher::generatePrefetch(Value v, unsigned opIdx, bool isPrologue,
                                   Attribute dotEncoding, OpBuilder &builder,
                                   llvm::Optional<int64_t> offsetK,
                                   llvm::Optional<int64_t> shapeK) {
  // opIdx: 0 => a, 1 => b
  auto type = v.getType().cast<RankedTensorType>();
  SmallVector<int64_t> shape{type.getShape().begin(), type.getShape().end()};
  SmallVector<int64_t> offset{0, 0};
  Type elementType = type.getElementType();

  auto intAttr = [&](int64_t val) { return builder.getI64IntegerAttr(val); };

  // k => (prefetchWidth, k - prefetchWidth)
  int64_t kIdx = opIdx == 0 ? 1 : 0;

  offset[kIdx] = isPrologue ? 0 : prefetchWidth;
  shape[kIdx] = isPrologue ? prefetchWidth : (shape[kIdx] - prefetchWidth);

  if (shapeK)
    shape[kIdx] = *shapeK;
  if (offsetK)
    offset[kIdx] = *offsetK;

  Value newSmem = builder.create<tensor::ExtractSliceOp>(
      v.getLoc(),
      // TODO: encoding?
      RankedTensorType::get(shape, elementType, type.getEncoding()), v,
      SmallVector<OpFoldResult>{intAttr(offset[0]), intAttr(offset[1])},
      SmallVector<OpFoldResult>{intAttr(shape[0]), intAttr(shape[1])},
      SmallVector<OpFoldResult>{intAttr(1), intAttr(1)});

  auto dotOperandEnc = triton::gpu::DotOperandEncodingAttr::get(
      builder.getContext(), opIdx, dotEncoding);
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

  // TODO: segfault (original for still has uses)
  // when used in flash attention that has 2 dots in the loop
  if (dotsInFor.size() > 1)
    return failure();

  // returns source of cvt
  auto getPrefetchSrc = [](Value v) -> Value {
    if (auto cvt = v.getDefiningOp<triton::gpu::ConvertLayoutOp>())
      if (isSharedEncoding(cvt.getOperand()))
        return cvt.src();
    return Value();
  };

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

  for (triton::DotOp dot : dotsInFor) {
    auto kSize = dot.a().getType().cast<RankedTensorType>().getShape()[1];
    // Skip prefetching if kSize is less than prefetchWidth
    if (kSize < prefetchWidth)
      continue;
    Value aSmem = getPrefetchSrc(dot.a());
    Value bSmem = getPrefetchSrc(dot.b());
    if (aSmem && bSmem) {
      Value aHeaderDef = getIncomingOp(aSmem);
      Value bHeaderDef = getIncomingOp(bSmem);
      // Only prefetch loop arg
      if (aHeaderDef && bHeaderDef) {
        dots.insert(dot);
        dot2aHeaderDef[dot] = aHeaderDef;
        dot2bHeaderDef[dot] = bHeaderDef;
        dot2aLoopArg[dot] = aSmem;
        dot2bLoopArg[dot] = bSmem;
        dot2aYield[dot] = getYieldOp(aSmem);
        dot2bYield[dot] = getYieldOp(bSmem);
      }
    }
  }

  return success();
}

void Prefetcher::emitPrologue() {
  OpBuilder builder(forOp);

  for (Value dot : dots) {
    Attribute dotEncoding =
        dot.getType().cast<RankedTensorType>().getEncoding();
    Value aPrefetched =
        generatePrefetch(dot2aHeaderDef[dot], 0, true, dotEncoding, builder);
    operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().a()] = aPrefetched;
    Value bPrefetched =
        generatePrefetch(dot2bHeaderDef[dot], 1, true, dotEncoding, builder);
    operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().b()] = bPrefetched;
  }
}

scf::ForOp Prefetcher::createNewForOp() {
  OpBuilder builder(forOp);

  SmallVector<Value> loopArgs;
  for (auto v : forOp.getIterOperands())
    loopArgs.push_back(v);
  for (Value dot : dots) {
    loopArgs.push_back(
        operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().a()]);
    loopArgs.push_back(
        operand2headPrefetch[dot.getDefiningOp<triton::DotOp>().b()]);
  }

  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), loopArgs);

  auto largestPow2 = [](int64_t n) -> int64_t {
    while ((n & (n - 1)) != 0)
      n = n & (n - 1);
    return n;
  };

  builder.setInsertionPointToStart(newForOp.getBody());
  BlockAndValueMapping mapping;
  for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
    mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);

  for (Operation &op : forOp.getBody()->without_terminator()) {
    Operation *newOp = builder.clone(op, mapping);
    auto dot = dyn_cast<triton::DotOp>(&op);
    if (dots.contains(dot)) {
      Attribute dotEncoding =
          dot.getType().cast<RankedTensorType>().getEncoding();
      // prefetched dot
      Operation *firstDot = builder.clone(*dot, mapping);
      if (Value a = operand2headPrefetch.lookup(dot.a()))
        firstDot->setOperand(
            0, newForOp.getRegionIterArgForOpOperand(*a.use_begin()));
      if (Value b = operand2headPrefetch.lookup(dot.b()))
        firstDot->setOperand(
            1, newForOp.getRegionIterArgForOpOperand(*b.use_begin()));

      // remaining part
      int64_t kOff = prefetchWidth;
      int64_t kRem = dot.a().getType().cast<RankedTensorType>().getShape()[1] -
                     prefetchWidth;
      Operation *prevDot = firstDot;
      while (kRem != 0) {
        int64_t kShape = largestPow2(kRem);
        Value aRem =
            generatePrefetch(mapping.lookup(dot2aLoopArg[dot]), 0, false,
                             dotEncoding, builder, kOff, kShape);
        Value bRem =
            generatePrefetch(mapping.lookup(dot2bLoopArg[dot]), 1, false,
                             dotEncoding, builder, kOff, kShape);
        newOp = builder.clone(*dot, mapping);
        newOp->setOperand(0, aRem);
        newOp->setOperand(1, bRem);
        newOp->setOperand(2, prevDot->getResult(0));
        prevDot = newOp;
        kOff += kShape;
        kRem -= kShape;
      }
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
    Attribute dotEncoding =
        dot.getType().cast<RankedTensorType>().getEncoding();
    yieldValues.push_back(generatePrefetch(mapping.lookup(dot2aYield[dot]), 0,
                                           true, dotEncoding, builder));
    yieldValues.push_back(generatePrefetch(mapping.lookup(dot2bYield[dot]), 1,
                                           true, dotEncoding, builder));
  }
  // Update ops of yield
  builder.create<scf::YieldOp>(yieldOp.getLoc(), yieldValues);
  return newForOp;
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
  }
};

} // anonymous namespace

std::unique_ptr<Pass> mlir::createTritonGPUPrefetchPass() {
  return std::make_unique<PrefetchPass>();
}
