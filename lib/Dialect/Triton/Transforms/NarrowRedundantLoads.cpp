#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-narrow-redundant-loads"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONNARROWREDUNDANTLOADS
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

// A load worth rewriting, together with the shape it should be narrowed to.
struct NarrowingCandidate {
  LoadOp load;
  SmallVector<int64_t> narrowShape;
};

// Re-materializes values at a reduced shape, which amounts to taking their
// slice at index 0 along every narrowed dimension. Only valid where the value
// is invariant along those dimensions, since any other slice would have done
// equally well.
class SliceMaterializer {
public:
  SliceMaterializer(OpBuilder &builder) : builder(builder) {}

  // Returns `v` restricted to `shape`, or failure if some operation in its
  // definition cannot be re-materialized at a narrower shape.
  FailureOr<Value> slice(Value v, ArrayRef<int64_t> shape);

  // Erases everything re-materialized so far. A narrowing that gives up
  // partway leaves behind an address computation nothing reads.
  void rollback() {
    for (Operation *op : llvm::reverse(created))
      op->erase();
    created.clear();
    cache.clear();
  }

private:
  static RankedTensorType narrowType(RankedTensorType type,
                                     ArrayRef<int64_t> shape) {
    return RankedTensorType::get(shape, type.getElementType(),
                                 type.getEncoding());
  }

  // Clones a single-result elementwise op with narrowed operands.
  FailureOr<Value> sliceElementwise(Operation *op, ArrayRef<int64_t> shape);

  // Records `op` so that `rollback` can undo it, and returns its result.
  Value track(Operation *op) {
    created.push_back(op);
    return op->getResult(0);
  }

  OpBuilder &builder;
  DenseMap<Value, Value> cache;
  SmallVector<Operation *> created;
};

// Returns true if `op` computes its result elementwise from operands of the
// same shape, so cloning it at a narrower shape computes the same slice.
// Constants are excluded because their value attribute is shaped, and so has
// to be rebuilt rather than carried over.
bool isNarrowableElementwise(Operation *op) {
  if (op->getNumResults() != 1 || op->getNumRegions() != 0)
    return false;
  if (isa<arith::ConstantOp>(op) || !isMemoryEffectFree(op))
    return false;
  if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
    return false;

  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return false;
  // A scalar operand contributes the same value to every lane, so only the
  // tensor operands have to line up with the result.
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    auto type = dyn_cast<RankedTensorType>(operand.getType());
    return !type || type.getShape() == resultType.getShape();
  });
}

// Returns true if `v` holds the same element in every lane.
bool isSplatLike(Value v) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return false;
  if (isa<SplatOp>(defOp))
    return true;
  auto constant = dyn_cast<arith::ConstantOp>(defOp);
  return constant && isa<SplatElementsAttr>(constant.getValue());
}

FailureOr<Value> SliceMaterializer::slice(Value v, ArrayRef<int64_t> shape) {
  auto type = dyn_cast<RankedTensorType>(v.getType());
  if (!type || type.getShape() == shape)
    return v;

  RankedTensorType resultType = narrowType(type, shape);
  if (Value cached = cache.lookup(v); cached && cached.getType() == resultType)
    return cached;

  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return failure();

  Location loc = defOp->getLoc();
  FailureOr<Value> result = failure();

  if (auto splat = dyn_cast<SplatOp>(defOp)) {
    result = track(SplatOp::create(builder, loc, resultType, splat.getSrc()));
  } else if (auto constant = dyn_cast<arith::ConstantOp>(defOp)) {
    if (auto splatAttr = dyn_cast<SplatElementsAttr>(constant.getValue())) {
      Value scalar = track(arith::ConstantOp::create(
          builder, loc, cast<TypedAttr>(splatAttr.getSplatValue<Attribute>())));
      result = track(SplatOp::create(builder, loc, resultType, scalar));
    }
  } else if (auto range = dyn_cast<MakeRangeOp>(defOp)) {
    // Keeping only the first element of a range collapses it to its start.
    if (shape.size() == 1 && shape[0] == 1) {
      Value start = track(arith::ConstantOp::create(
          builder, loc, builder.getI32IntegerAttr(range.getStart())));
      result = track(SplatOp::create(builder, loc, resultType, start));
    }
  } else if (auto broadcast = dyn_cast<BroadcastOp>(defOp)) {
    // The source is already unit-sized along the dimensions it broadcasts, so
    // narrowing it to the same shape leaves those dimensions alone.
    auto srcType = cast<RankedTensorType>(broadcast.getSrc().getType());
    SmallVector<int64_t> srcShape;
    for (auto [srcDim, dim] : llvm::zip(srcType.getShape(), shape))
      srcShape.push_back(std::min(srcDim, dim));
    FailureOr<Value> src = slice(broadcast.getSrc(), srcShape);
    if (succeeded(src)) {
      result = src->getType() == resultType
                   ? *src
                   : track(BroadcastOp::create(builder, loc, resultType, *src));
    }
  } else if (auto expand = dyn_cast<ExpandDimsOp>(defOp)) {
    // The expanded dimension is unit-sized in the result, hence unit-sized in
    // any narrowing of it, so it can be dropped and re-inserted unchanged.
    uint32_t axis = expand.getAxis();
    SmallVector<int64_t> srcShape(shape);
    srcShape.erase(srcShape.begin() + axis);
    FailureOr<Value> src = slice(expand.getSrc(), srcShape);
    if (succeeded(src))
      result =
          track(ExpandDimsOp::create(builder, loc, resultType, *src, axis));
  } else if (auto addPtr = dyn_cast<AddPtrOp>(defOp)) {
    FailureOr<Value> ptr = slice(addPtr.getPtr(), shape);
    FailureOr<Value> offset = slice(addPtr.getOffset(), shape);
    if (succeeded(ptr) && succeeded(offset))
      result = track(AddPtrOp::create(builder, loc, resultType, *ptr, *offset));
  } else if (isNarrowableElementwise(defOp)) {
    result = sliceElementwise(defOp, shape);
  }

  if (succeeded(result))
    cache[v] = *result;
  return result;
}

FailureOr<Value> SliceMaterializer::sliceElementwise(Operation *op,
                                                     ArrayRef<int64_t> shape) {
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    FailureOr<Value> narrowed = slice(operand, shape);
    if (failed(narrowed))
      return failure();
    operands.push_back(*narrowed);
  }

  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(
      narrowType(cast<RankedTensorType>(op->getResult(0).getType()), shape));
  state.addAttributes(op->getAttrs());
  return track(builder.create(state));
}

// Returns the shape `load` can be narrowed to, if any. A dimension qualifies
// when the analysis proves the loaded values repeat across its whole extent.
// That constancy is the gcd of the address and mask constancies; `other` is
// checked separately because it is not part of it.
std::optional<SmallVector<int64_t>>
getNarrowShape(LoadOp load, ModuleAxisInfoAnalysis &axisInfo) {
  auto type = dyn_cast<RankedTensorType>(load.getType());
  if (!type || !type.hasStaticShape())
    return std::nullopt;
  // Dropping reads of a volatile load would change how many times it is issued.
  if (load.getIsVolatile())
    return std::nullopt;
  if (load.getOther() && !isSplatLike(load.getOther()))
    return std::nullopt;

  AxisInfo *info = axisInfo.getAxisInfo(load.getResult());
  if (!info || info->getRank() != type.getRank())
    return std::nullopt;

  ArrayRef<int64_t> shape = type.getShape();
  SmallVector<int64_t> narrowShape(shape);
  bool narrowed = false;
  for (auto [dim, extent] : llvm::enumerate(shape)) {
    if (extent > 1 && info->getConstancy(dim) == extent) {
      narrowShape[dim] = 1;
      narrowed = true;
    }
  }
  return narrowed ? std::optional(narrowShape) : std::nullopt;
}

LogicalResult narrowLoad(const NarrowingCandidate &candidate) {
  LoadOp load = candidate.load;
  IRRewriter rewriter(load);
  SliceMaterializer materializer(rewriter);

  // Narrow every operand before building the load, so that giving up on one of
  // them leaves the original load in place.
  SmallVector<Value> operands;
  for (Value operand : {load.getPtr(), load.getMask(), load.getOther()}) {
    if (!operand) {
      operands.push_back(nullptr);
      continue;
    }
    FailureOr<Value> narrowed =
        materializer.slice(operand, candidate.narrowShape);
    if (failed(narrowed)) {
      materializer.rollback();
      return failure();
    }
    operands.push_back(*narrowed);
  }

  auto type = cast<RankedTensorType>(load.getType());
  auto narrowedType = RankedTensorType::get(
      candidate.narrowShape, type.getElementType(), type.getEncoding());
  Value narrowedLoad =
      LoadOp::create(rewriter, load.getLoc(), narrowedType, operands[0],
                     operands[1], operands[2], load.getCacheAttr(),
                     load.getEvictAttr(), load.getIsVolatileAttr());
  rewriter.replaceOpWithNewOp<BroadcastOp>(load, type, narrowedLoad);
  return success();
}

} // namespace

class NarrowRedundantLoadsPass
    : public impl::TritonNarrowRedundantLoadsBase<NarrowRedundantLoadsPass> {
public:
  void runOnOperation() override {
    ModuleOp m = getOperation();
    ModuleAxisInfoAnalysis axisInfo(m);

    // Collect first: rewriting the loads invalidates the analysis.
    SmallVector<NarrowingCandidate> candidates;
    m.walk([&](LoadOp load) {
      if (std::optional<SmallVector<int64_t>> narrowShape =
              getNarrowShape(load, axisInfo))
        candidates.push_back({load, std::move(*narrowShape)});
    });

    for (const NarrowingCandidate &candidate : candidates) {
      // Logged before the rewrite, which erases the load.
      LLVM_DEBUG(llvm::dbgs() << "narrowing " << candidate.load << "\n");
      if (failed(narrowLoad(candidate)))
        LLVM_DEBUG(llvm::dbgs() << "declined: address not re-materializable\n");
    }
  }
};

} // namespace mlir::triton
