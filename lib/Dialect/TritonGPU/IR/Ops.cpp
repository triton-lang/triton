#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/DebugStringHelper.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#define GET_OP_CLASSES
#include "triton/Dialect/TritonGPU/IR/Ops.cpp.inc"

namespace mlir::triton::gpu {

namespace {

template <typename T> bool hasEncoding(Value value) {
  auto type = value.getType();
  if (auto tensorType = dyn_cast<TensorOrMemDesc>(type)) {
    auto encoding = tensorType.getEncoding();
    return encoding && isa<T>(encoding);
  }
  return false;
}

bool hasDotOperandEncoding(Value value) {
  return hasEncoding<triton::gpu::DotOperandEncodingAttr>(value);
}

bool isConvertTrivial(ConvertLayoutOp op) {
  auto srcType = op.getSrc().getType();
  auto dstType = op.getType();
  auto srcEncoding = srcType.getEncoding();
  auto dstEncoding = dstType.getEncoding();
  return cast<DialectInferLayoutInterface>(&srcEncoding.getDialect())
      ->verifyLayoutsAreEqual(srcType.getShape(), srcEncoding, dstEncoding, {})
      .succeeded();
}

} // namespace

//===----------------------------------------------------------------------===//
// Canonicalizer
//===----------------------------------------------------------------------===//

// reshape(cvt) -> reshape
struct CanonicalizeConvertFromReshape
    : public mlir::OpRewritePattern<triton::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::ReshapeOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    // If the layouts are structurally the same, the convert is trivial
    if (isConvertTrivial(convert)) {
      rewriter.replaceOpWithNewOp<triton::ReshapeOp>(
          op, op.getType(), convert.getSrc(), op.getAllowReorder(),
          op.getEfficientLayout());
      return success();
    }

    if (isExpensiveView(convert.getSrc().getType(), op.getType()))
      return failure();
    if (!op.getAllowReorder())
      return failure();

    rewriter.replaceOpWithNewOp<triton::ReshapeOp>(
        op, op.getType(), convert.getSrc(), op.getAllowReorder(),
        op.getEfficientLayout());
    return mlir::success();
  }
};

// TODO We should do this generically for op(cvt) -> op
// We have similar patterns for reshape and split...
// See https://github.com/triton-lang/triton/pull/5403#discussion_r1920091671

// trans(cvt) -> trans
struct CanonicalizeConvertFromTranspose
    : public mlir::OpRewritePattern<triton::TransOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::TransOp op,
                  PatternRewriter &rewriter) const override {
    // transpose(x, order=[0, 1, ...]) -> x
    // We turn it into a (trivial) convert_layout that may be folded away
    if (isIota(op.getOrder())) {
      rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, op.getType(),
                                                   op.getSrc());
      return success();
    }

    // If the layouts are structurally the same, the convert is trivial
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert || !isConvertTrivial(convert))
      return failure();

    rewriter.replaceOpWithNewOp<triton::TransOp>(
        op, op.getType(), convert.getSrc(), op.getOrder());
    return success();
  }
};

// histogram(cvt) -> histogram
struct CanonicalizeConvertFromHistogram
    : public mlir::OpRewritePattern<triton::HistogramOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::HistogramOp op,
                  PatternRewriter &rewriter) const override {
    auto src = op.getSrc();
    auto convert = src.getDefiningOp<ConvertLayoutOp>();
    if (!convert) {
      return failure();
    }
    src = convert.getSrc();

    // If mask is present, convert the layout of mask to match new src layout
    auto mask = op.getMask();
    if (mask) {
      auto sharedType = getI1SameShape(src.getType());
      rewriter.setInsertionPoint(op);
      mask = rewriter.create<ConvertLayoutOp>(op.getLoc(), sharedType, mask);
    }

    rewriter.replaceOpWithNewOp<triton::HistogramOp>(
        op, op->getResult(0).getType(), src, mask);
    return success();
  }
};

// If the gather does not have an optimized layout attached, then the source
// layout does not matter since the gather will be codegen'd by storing the
// source tensor into shared memory. Thus, we can fold conversions into the
// source operand.
//
// gather(cvt(src), idx) -> gather(src, idx)
struct CanonicalizeConvertFromGatherSource : public OpRewritePattern<GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(GatherOp op, PatternRewriter &rewriter) const override {
    // Don't do this if the compiler picked an optimized layout.
    if (op.getEfficientLayout())
      return failure();

    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();

    rewriter.replaceOpWithNewOp<GatherOp>(op, convert.getSrc(), op.getIndices(),
                                          op.getAxis());
    return success();
  }
};

// alloc(cvt) -> alloc
struct CanonicalizeConvertFromAlloc
    : public mlir::OpRewritePattern<triton::gpu::LocalAllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::gpu::LocalAllocOp op,
                  PatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return failure();
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    rewriter.replaceOpWithNewOp<triton::gpu::LocalAllocOp>(
        op, op->getResult(0).getType(), convert.getSrc());
    return mlir::success();
  }
};

// local_store(cvt) -> local_store
struct CanonicalizeConvertFromLocalStore
    : public mlir::OpRewritePattern<triton::gpu::LocalStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::gpu::LocalStoreOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    rewriter.replaceOpWithNewOp<triton::gpu::LocalStoreOp>(op, convert.getSrc(),
                                                           op.getDst());
    return mlir::success();
  }
};

struct CanonicalizeConvertFromSplit
    : public mlir::OpRewritePattern<triton::SplitOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::SplitOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    auto srcEncoding = convert.getSrc().getType().getEncoding();
    // Multiple source layout can give the same output layout, if the source
    // layout of the convert gives the same destination layout we can skip the
    // convert.
    auto dstEncoding = inferDstEncoding(op, srcEncoding);
    if (dstEncoding != op.getOutLHS().getType().getEncoding())
      return failure();
    rewriter.replaceOpWithNewOp<triton::SplitOp>(op, convert.getSrc());
    return mlir::success();
  }
};

struct CanonicalizeConvertFromConvert
    : public OpRewritePattern<ConvertLayoutOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ConvertLayoutOp op,
                  PatternRewriter &rewriter) const override {
    // Convert to the same layout is redundant.
    if (op->getResultTypes() == op->getOperandTypes()) {
      rewriter.replaceOp(op, op->getOperands());
      return success();
    }

    // We don't handle conversions to DotOperandEncodingAttr.  This is a
    // heuristic to accommodate fused attention.
    auto srcType = op.getSrc().getType();
    auto dstType = op.getType();
    if (mlir::isa<DotOperandEncodingAttr>(dstType.getEncoding()) &&
        mlir::isa<NvidiaMmaEncodingAttr>(srcType.getEncoding()))
      return failure();

    Operation *arg = op.getSrc().getDefiningOp();
    if (!arg)
      return failure();

    // cvt(reshape) -> reshape
    if (auto reshape = dyn_cast<ReshapeOp>(arg)) {
      if (!reshape.getAllowReorder() || reshape.getEfficientLayout() ||
          isExpensiveView(reshape.getSrc().getType(), op.getType()))
        return failure();

      // In TritonGPUToLLVM phase, ViewOp is converted to unpacking and packing
      // operations, which requires the element type to match between unpacking
      // and packing. However, part of values with dot operand encoding will be
      // packed/unpacked as i32 elements instead of the underlying element type.
      // To avoid errors, skip this folding when either the operand or result
      // of view has a dot operand encoding.
      if (hasDotOperandEncoding(op->getOperand(0)) ||
          hasDotOperandEncoding(op->getResult(0)))
        return failure();

      rewriter.replaceOpWithNewOp<ReshapeOp>(op, op->getResult(0).getType(),
                                             reshape.getResult(),
                                             reshape.getAllowReorder());
      return success();
    }

    // cvt(histogram) -> histogram
    if (auto histogram = dyn_cast<HistogramOp>(arg)) {
      // For histogram ops the input and output layouts are independent, so we
      // can always fold convert into the histogram op.
      rewriter.replaceOpWithNewOp<HistogramOp>(op, op->getResult(0).getType(),
                                               histogram.getSrc(),
                                               histogram.getMask());
      return success();
    }

    // cvt(local_load) -> local_load.
    if (auto sharedLoad = dyn_cast<LocalLoadOp>(arg)) {
      // Shared_load can load to any layout so we can always fold convert into
      // it.
      // We insert at the point of the original op as there could be ops with
      // memory side-effects between the LocalLoad op and the ConvertLayout op
      rewriter.setInsertionPoint(arg);
      rewriter.replaceOpWithNewOp<LocalLoadOp>(op, op->getResult(0).getType(),
                                               sharedLoad.getSrc(),
                                               sharedLoad.getToken());

      return success();
    }

    // cvt(cat) -> cat
    if (auto cat = dyn_cast<CatOp>(arg)) {
      if (isExpensiveCat(cat, op.getType().getEncoding()))
        return failure();

      rewriter.replaceOpWithNewOp<CatOp>(op, op->getResult(0).getType(),
                                         cat.getOperands());
      return success();
    }

    // cvt(cvt(x, type1), type2) -> cvt(x, type2)
    if (auto cvt = dyn_cast<ConvertLayoutOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
          op, op->getResultTypes().front(), cvt.getSrc());
      return success();
    }

    // cvt(type1, splat(type2, x)) -> splat(type1, x)
    if (auto splat = dyn_cast<triton::SplatOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op->getResultTypes(),
                                                   splat.getSrc());
      return success();
    }

    // cvt(type1, make_range(type2, x)) -> make_range(type1, x)
    if (auto range = dyn_cast<MakeRangeOp>(arg)) {
      rewriter.replaceOpWithNewOp<MakeRangeOp>(
          op, op->getResultTypes(), range.getStart(), range.getEnd());
      return success();
    }

    // cvt(type, constant) -> constant
    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(arg))
      if (auto ret = dyn_cast<SplatElementsAttr>(cst.getValue())) {
        auto ty = cast<ShapedType>(op->getResultTypes().front());
        auto newRet =
            SplatElementsAttr::get(ty, ret.getSplatValue<Attribute>());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newRet);
        return success();
      }
    return failure();
  }
};

void ConvertLayoutOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<CanonicalizeConvertFromConvert>(context);
  patterns.add<CanonicalizeConvertFromReshape>(context);
  patterns.add<CanonicalizeConvertFromTranspose>(context);
  patterns.add<CanonicalizeConvertFromGatherSource>(context);
  patterns.add<CanonicalizeConvertFromHistogram>(context);
  patterns.add<CanonicalizeConvertFromAlloc>(context);
  patterns.add<CanonicalizeConvertFromLocalStore>(context);
  patterns.add<CanonicalizeConvertFromSplit>(context);
}

LogicalResult Fp4ToFpOp::verify() {
  auto srcTy = cast<RankedTensorType>(getSrc().getType());
  auto resTy = cast<RankedTensorType>(getResult().getType());
  auto rank = srcTy.getRank();

  if (rank != resTy.getRank())
    return emitError() << "source rank " << rank << " != result rank "
                       << resTy.getRank();

  auto srcShape = srcTy.getShape();
  auto resShape = resTy.getShape();
  auto axis = getAxis();

  if (!(0 <= axis && axis < rank))
    return emitError() << "axis " << axis << " out of range for rank " << rank;

  auto elemType = resTy.getElementType();
  if (!(elemType.isBF16() || elemType.isF16()))
    return emitError() << "only bf16 or f16 is supported for now, got "
                       << elemType;

  for (int i = 0; i < rank; ++i) {
    if (i == axis) {
      if (resShape[i] != srcShape[i] * 2)
        return emitError() << "axis " << axis
                           << " dimension must be 2x source dimension (src="
                           << srcShape[i] << ", dst=" << resShape[i] << ")";
    } else {
      if (resShape[i] != srcShape[i])
        return emitError() << "dimension " << i
                           << " mismatch (src=" << srcShape[i]
                           << ", dst=" << resShape[i] << ", axis=" << axis
                           << ")";
    }
  }
  return success();
}

void Fp4ToFpOp::build(OpBuilder &builder, OperationState &state,
                      TypedValue<RankedTensorType> src, Type elemType,
                      int32_t axis) {
  auto srcTy = src.getType();
  auto shape = llvm::to_vector(srcTy.getShape());
  auto rank = srcTy.getRank();
  assert(0 <= axis && axis < rank);
  shape[axis] *= 2;

  Attribute inEnc = srcTy.getEncoding();
  Attribute outEnc;
  auto result =
      inEnc.getDialect()
          .getRegisteredInterface<triton::DialectInferLayoutInterface>()
          ->inferFp4ToFpOpEncoding(shape, axis, inEnc, outEnc,
                                   /*fwdInference=*/true, state.location);
  assert(succeeded(result));

  auto resultTy = RankedTensorType::get(shape, elemType, outEnc);
  build(builder, state, resultTy, src, axis);
}

OpFoldResult MemDescTransOp::fold(FoldAdaptor adaptor) {
  // transpose(x, order=[0, 1, ...]) -> x
  if (isIota(getOrder())) {
    return getSrc();
  }

  // transpose(transpose(x)) -> transpose(x)
  if (auto innerTrans = getSrc().getDefiningOp<MemDescTransOp>()) {
    setOrder(applyPermutation(innerTrans.getOrder(), getOrder()));
    setOperand(innerTrans.getSrc());
    return getResult();
  }

  return {};
}

LogicalResult
MemDescTransOp::inferReturnTypes(MLIRContext *context,
                                 std::optional<Location> loc,
                                 MemDescTransOp::Adaptor adaptor,
                                 SmallVectorImpl<Type> &inferredReturnTypes) {

  // type is the same as the input
  auto argTy = cast<MemDescType>(adaptor.getSrc().getType());
  auto shape = argTy.getShape();
  auto order = adaptor.getOrder();
  SmallVector<int64_t> retShape = applyPermutation(shape, order);

  auto retEltTy = argTy.getElementType();
  Attribute argEncoding = argTy.getEncoding();
  Attribute retEncoding;
  if (argEncoding) {
    Dialect &dialect = argEncoding.getDialect();
    auto inferLayoutInterface = cast<DialectInferLayoutInterface>(&dialect);
    if (failed(inferLayoutInterface->inferTransOpEncoding(
            argEncoding, shape, order, retEncoding, loc))) {
      return failure();
    }
  }

  // Permute the last `rank` dims of the source alloc shape.
  SmallVector<int64_t> allocShape =
      applyPermutation(argTy.getAllocShape().take_back(order.size()), order);
  allocShape.insert(allocShape.begin(), argTy.getAllocShape().begin(),
                    argTy.getAllocShape().end() - order.size());

  inferredReturnTypes.push_back(
      MemDescType::get(retShape, retEltTy, retEncoding, argTy.getMemorySpace(),
                       argTy.getMutableMemory(), allocShape));
  return success();
}

// MemDescReshapeOp
LogicalResult MemDescReshapeOp::verify() {
  MemDescType dstType = getResult().getType();
  MemDescType srcType = getSrc().getType();
  if (product(dstType.getShape()) != product(srcType.getShape())) {
    return emitError(
        "number of src and dst elements of reshape must be the same");
  }
  if (dstType.getElementType() != srcType.getElementType()) {
    return emitError("result element type must match src element type");
  }
  auto srcShape = srcType.getShape();
  if (srcType.getAllocShape().take_back(srcShape.size()) != srcShape) {
    return emitError("NYI: memdesc_reshape of memdesc_subviews");
  }

  MemDescType expectedTy;
  if (failed(inferReturnTypes(getContext(), getLoc(), srcType,
                              dstType.getShape(), expectedTy)))
    return failure();
  // Check that the alloc shape separately to give a cleaner error, given that
  // it's the most likely source of the error.
  if (expectedTy.getAllocShape() != dstType.getAllocShape()) {
    return emitError(
        "The result alloc shape does not match the expected alloc shape.");
  }
  if (expectedTy != dstType) {
    return emitError("source and destination layout are incompatible.");
  }
  return success();
}

static LogicalResult inferMemDescReshapeOpEncoding(ArrayRef<int64_t> srcShape,
                                                   Attribute srcEnc,
                                                   ArrayRef<int64_t> dstShape,
                                                   Attribute &dstEnc) {
  if (auto mmaEncoding = dyn_cast<NVMMASharedEncodingAttr>(srcEnc)) {
    // TODO: supporting reshape of CTA layouts is non-trivial.
    if (getNumCTAs(mmaEncoding) > 1)
      return failure();
    int innerDimDst =
        mmaEncoding.getTransposed() ? dstShape.front() : dstShape.back();
    int innerDimSrc =
        mmaEncoding.getTransposed() ? srcShape.front() : srcShape.back();
    // For now disallow reshape of the inner dimension.
    if (innerDimDst != innerDimSrc)
      return failure();
    auto *ctx = srcEnc.getContext();

    // CTALayout can be all 1's because we bailed on multi-CTA layouts above.
    auto CTALayout = CTALayoutAttr::get(
        ctx,
        /*CTAsPerCGA=*/SmallVector<unsigned>(dstShape.size(), 1),
        /*CTASplitNum=*/SmallVector<unsigned>(dstShape.size(), 1),
        /*CTAOrder=*/llvm::to_vector(llvm::seq<unsigned>(dstShape.size())));
    dstEnc = NVMMASharedEncodingAttr::get(
        ctx, mmaEncoding.getSwizzlingByteWidth(), mmaEncoding.getTransposed(),
        mmaEncoding.getElementBitWidth(), mmaEncoding.getFp4Padded(),
        CTALayout);
    // Big guns, check linear layouts are equivalent
    // We disallow reshaping memdesc_subviews in the verifier
    // We disallow reshaping memdesc_subviews in the verifier
    auto srcLL = toLinearLayout(srcShape, srcEnc, srcShape);
    auto dstLL = toLinearLayout(dstShape, dstEnc, dstShape);
    if (reshapeLayout(ctx, srcLL, dstShape) != dstLL) {
      return failure();
    }
    return success();
  }
  return failure();
}

LogicalResult MemDescReshapeOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> loc, MemDescType srcTy,
    ArrayRef<int64_t> dstShape, MemDescType &inferredReturnType) {
  if (product<int64_t>(dstShape) != product<int64_t>(srcTy.getShape()))
    return emitOptionalError(
        loc, "dst shape has different number of elements than src");

  Attribute dstEncoding;
  if (Attribute srcEnc = srcTy.getEncoding()) {
    if (failed(inferMemDescReshapeOpEncoding(srcTy.getShape(), srcEnc, dstShape,
                                             dstEncoding)))
      return failure();
  }

  SmallVector<int64_t> dstAllocShape =
      to_vector(srcTy.getAllocShape().take_front(srcTy.getAllocShape().size() -
                                                 srcTy.getShape().size()));
  dstAllocShape.append(dstShape.begin(), dstShape.end());

  inferredReturnType = MemDescType::get(
      dstShape, srcTy.getElementType(), dstEncoding, srcTy.getMemorySpace(),
      srcTy.getMutableMemory(), dstAllocShape);
  return success();
}

// MemDescReinterpretOp
LogicalResult MemDescReinterpretOp::verify() {
  if (getSrc().getType().getMemorySpace() != getType().getMemorySpace())
    return emitError("source and destination memory space must match");
  return success();
}

OpFoldResult MemDescReinterpretOp::fold(FoldAdaptor adaptor) {
  if (getType() == getSrc().getType())
    return getSrc();
  return {};
}

// LocalAllocOp
void LocalAllocOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  Operation *op = getOperation();
  // If allocation is immutable, mark it as no side effect allow things like
  // CSE, DCE to work in early compiler passes.
  // After the memory offset is computed, we attach the true side effect to the
  // op.
  if (!getType().getMutableMemory() && !op->hasAttr("allocation.offset"))
    return;
  OpResult alloc = getOperation()->getOpResult(0);
  effects.emplace_back(MemoryEffects::Allocate::get(), alloc,
                       SharedMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(), alloc,
                         SharedMemory::get());
}

OpFoldResult LocalAllocOp::fold(FoldAdaptor adaptor) {
  if (getType().getMutableMemory())
    return {};
  auto src = getSrc();
  if (!src)
    return {};
  auto localLoadOp = src.getDefiningOp<LocalLoadOp>();
  if (!localLoadOp)
    return {};
  auto loadSrc = localLoadOp.getSrc();
  if (loadSrc.getType() != getType())
    return {};
  return loadSrc;
}

LogicalResult verifyMemoryOpTypes(Operation *op, ShapedType srcTy,
                                  ShapedType dstTy) {
  if (srcTy.getElementType() != dstTy.getElementType()) {
    return op->emitOpError("source element type ")
           << srcTy << " must match "
           << "destination element type " << dstTy.getElementType();
  }
  if (srcTy.getShape() != dstTy.getShape()) {
    return op->emitOpError("source shape [")
           << srcTy.getShape() << "] must match ["
           << "destination shape " << dstTy.getShape() << "]";
  }
  return success();
}

LogicalResult verifyAllocOp(Operation *op, Value src, MemDescType dstTy) {
  if (dstTy.getShape() != dstTy.getAllocShape())
    return op->emitOpError("result shape and its alloc shape must match");

  if (!src) {
    if (!dstTy.getMutableMemory()) {
      return op->emitOpError(
          "uninitialized alloc must have a mutable memdesc type");
    }
    return success();
  }

  return verifyMemoryOpTypes(op, cast<RankedTensorType>(src.getType()), dstTy);
}

LogicalResult LocalAllocOp::verify() {
  if (!isa<SharedMemorySpaceAttr>(getType().getMemorySpace()))
    return emitOpError("should create a buffer of shared memory");

  return verifyAllocOp(*this, getSrc(), getType());
}

// LocalStoreOp
LogicalResult LocalStoreOp::verify() {
  if (!getDst().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return verifyMemoryOpTypes(*this, getSrc().getType(), getDst().getType());
}

// LocalLoadOp
LogicalResult LocalLoadOp::verify() {
  return verifyMemoryOpTypes(*this, getSrc().getType(), getType());
}

// AsyncCopyGlobalToLocalOp
LogicalResult AsyncCopyGlobalToLocalOp::verify() {
  if (!getResult().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return success();
}

LogicalResult MemDescSubviewOp::verify() {
  auto srcTy = getSrc().getType();
  auto dstTy = getType();

  if (srcTy.getElementType() != dstTy.getElementType()) {
    return emitError("result element type must match desc element type");
  }
  if (getOffsets().size() != srcTy.getRank()) {
    return emitError("offsets must have the same rank as input");
  }
  if (srcTy.getRank() < dstTy.getRank()) {
    return emitError("result rank must be less than or equal to input rank");
  }
  auto rankDiff = srcTy.getRank() - dstTy.getRank();
  for (int i = 0; i < dstTy.getRank(); i++) {
    if (dstTy.getDimSize(i) > srcTy.getDimSize(i + rankDiff)) {
      return emitError(
                 "result shape cannot be larger than input shape at dimension ")
             << i;
    }
  }

  auto srcEnc = srcTy.getEncoding();
  auto dstEnc = dstTy.getEncoding();
  if (!!srcEnc != !!dstEnc) {
    return emitError("src and result must both have or not have an encoding");
  }

  if (!isa<SharedEncodingTrait>(srcEnc) &&
      !isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(srcEnc)) {
    return emitError("src encoding must be SharedEncodingTrait");
  }
  if (!isa<SharedEncodingTrait>(dstEnc) &&
      !isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(srcEnc)) {
    return emitError("result encoding must be SharedEncodingTrait");
  }

  if (isa<triton::nvidia_gpu::TensorMemoryEncodingAttr>(srcEnc)) {
    // We support only 3D -> 2D subviews with only first offset being non-zero.
    if (srcTy.getRank() != 3 || dstTy.getRank() != 2) {
      return emitError("only 3D -> 2D subviews are supported for "
                       "TensorMemoryEncodingAttr");
    }
    for (int i = 1; i < srcTy.getRank(); i++) {
      if (auto constOp = getOffsets()[i].getDefiningOp<arith::ConstantOp>()) {
        if (!isa<IntegerAttr>(constOp.getValue()) ||
            cast<IntegerAttr>(constOp.getValue()).getInt() != 0) {
          return emitError("only first offset can be non-zero for the subview"
                           "of TensorMemoryEncodingAttr");
        }
      } else {
        return emitError(
            "offsets other than the first one must be constant zeros");
      }
    }
    return success();
  }

  assert(isa<SharedEncodingTrait>(srcEnc));

  // corner case: 1D -> 1D into a 1 element tensor (we don't have 0D tensors)
  if (srcTy.getRank() == 1 && dstTy.getRank() == 1 &&
      dstTy.getDimSize(0) == 1) {
    return success();
  }

  // There are two cases:
  // 1. The subview is rank-reducing
  //  - We split along the first dimension. It can be with non-constant offsets
  if (srcTy.getRank() != dstTy.getRank()) {
    if (srcTy.getRank() - dstTy.getRank() != 1) {
      return emitError(
          "only nD -> (n-1)D rank-reducing subviews are supported");
    }
    for (auto offset : getOffsets().take_back(dstTy.getRank())) {
      APInt value;
      if (!matchPattern(offset, m_ConstantInt(&value))) {
        return emitError("only constant values are allowed outside the front "
                         "dimension in a rank-reducing subview");
      }
      if (!value.isZero()) {
        return emitError(
            "only first offset can be non-zero for a rank-reducing subview");
      }
    }
    return success();
  }
  assert(srcTy.getRank() == dstTy.getRank());
  // 2. The src is non-rank-reducing
  //  - We split along at most one dim, but just with constant values
  //  - The values where the split happens must not be within the swizzling
  //  pattern
  // Check which dimensions we are splitting along
  SetVector<int> splitDims{};
  for (int i = 0; i < srcTy.getRank(); i++) {
    if (srcTy.getDimSize(i) != dstTy.getDimSize(i)) {
      splitDims.insert(i);
    }
  }
  SmallVector<int64_t> offsets;
  for (auto offset : getOffsets()) {
    APInt value;
    if (!matchPattern(offset, m_ConstantInt(&value)))
      return emitError("only constant values are allowed for the split");
    offsets.push_back(value.getSExtValue());
  }
  // Identity subview
  if (splitDims.empty()) {
    return success();
  }

  for (auto [dim, offset] : llvm::enumerate(offsets)) {
    if (!splitDims.contains(dim)) {
      if (offset != 0) {
        return emitError("A non zero offset found in a dimension that is "
                         "not being split");
      }
    } else {
      if (offset & (dstTy.getDimSize(dim) - 1)) {
        return emitError("The split offset may not touch the tile");
      }
    }
  }

  auto ctx = getContext();
  auto ll = triton::gpu::toLinearLayout(srcTy);
  // NYI: We don't support non-trivial block dimension for now.
  auto kBlock = mlir::StringAttr::get(getContext(), "block");
  if (ll.getInDimSize(kBlock) != 1) {
    return emitError("non-trivial block dimension not supported");
  }

  auto llInv = ll.invert();
  for (auto dim : splitDims) {
    auto kDim = mlir::StringAttr::get(ctx, "dim" + llvm::Twine(dim));
    llvm::SmallVector<std::pair<mlir::StringAttr, int32_t>> namedOffsets;
    for (auto d : standardOutDimNames(ctx, srcTy.getRank())) {
      namedOffsets.push_back({d, 0});
    }
    for (int dimSize = dstTy.getDimSize(dim); dimSize < srcTy.getDimSize(dim);
         dimSize *= 2) {
      namedOffsets[dim] = {kDim, dimSize};
      if (!llvm::isPowerOf2_32(llInv.apply(namedOffsets)[0].second)) {
        return emitError(
            "We don't support splitting along the swizzling pattern");
      }
    }
  }
  return success();
}

// -- LocalAllocOp --

int32_t LocalAllocOp::getAlignmentOrDefault() {
  auto align = getAlignment();
  if (align) {
    return *align;
  }

  auto ty = getType();
  auto enc = dyn_cast<SharedEncodingTrait>(ty.getEncoding());
  return enc ? enc.getAlignment() : 16;
}

// -- WarpSpecializeOp --

static Type removeEncodingIfTensor(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return tensorType.cloneWithEncoding({});
  }
  return type;
}

RegionRange WarpSpecializeOp::getPartitionRegions() {
  return cast<WarpSpecializePartitionsOp>(
             getPartitionOpHolder().front().front())
      .getPartitionRegions();
}

void WarpSpecializeOp::getSuccessorRegions(
    RegionBranchPoint src, SmallVectorImpl<RegionSuccessor> &successors) {
  // The parent branches transparently into the default region.
  if (src.isParent()) {
    successors.emplace_back(&getDefaultRegion());
    return;
  }
  // And the default region branches transparently back to the parent.
  assert(src.getRegionOrNull() == &getDefaultRegion());
  successors.push_back(RegionSuccessor(getResults()));
}

LogicalResult WarpSpecializeOp::verify() {
  // The default region is not isolated from above but the partition regions
  // have to be. MLIR does not support this, so we hide an op inside another
  // region that contains the isolated regions. Check that it is there.
  if (!isa<WarpSpecializePartitionsOp>(
          getPartitionOpHolder().front().front())) {
    return emitOpError(
        "expected to find only a `ttg.warp_specialize.partitions` op inside "
        "its second region");
  }

  // Verify the partitions.
  if (getPartitionRegions().size() != getPartitionNumWarps().size()) {
    return emitOpError("has ") << getPartitionRegions().size()
                               << " partitions but `partitionNumWarps` has "
                               << getPartitionNumWarps().size() << " elements";
  }
  for (auto [i, numWarps] : llvm::enumerate(getPartitionNumWarps())) {
    if (llvm::isPowerOf2_32(numWarps))
      continue;
    return emitOpError("partition #")
           << i << " number of warps (" << numWarps << ") must be a power of 2";
  }
  if (std::optional<ArrayRef<int32_t>> startIds = getWarpGroupStartIds()) {
    if (startIds->size() != getPartitionNumWarps().size()) {
      return emitOpError("has ")
             << startIds->size() << " warp group start IDs but expected "
             << getPartitionNumWarps().size();
    }
  }

  for (auto [i, region] : llvm::enumerate(getPartitionRegions())) {
    if (region->getNumArguments() != getNumOperands()) {
      return emitOpError("partition region #")
             << i << " has " << region->getNumArguments()
             << " arguments but expected " << getNumOperands();
    }
    for (auto [argIdx, argType, capType] : llvm::enumerate(
             region->getArgumentTypes(), getExplicitCaptures().getTypes())) {
      if (argType == capType)
        continue;
      return emitOpError("partition region #")
             << i << " argument #" << argIdx << " has type " << argType
             << " but corresponding capture has type " << capType;
    }
  }

  // This op cannot be nested inside itself.
  if ((*this)->getParentOfType<WarpSpecializeOp>()) {
    return emitOpError(
        "cannot be nested inside another `ttg.warp_specialize` op");
  }

  return success();
}

LogicalResult WarpSpecializeOp::canonicalize(WarpSpecializeOp op,
                                             PatternRewriter &b) {
  // Propagate unused results and captures by removing them from the op.
  llvm::BitVector unusedArgs(op.getNumOperands());
  llvm::BitVector unusedResults(op.getNumResults());
  for (auto [i, result] : llvm::enumerate(op.getResults())) {
    if (result.use_empty())
      unusedResults.set(i);
  }
  // Remove duplicate captures.
  DenseMap<Value, unsigned> uniqueCaptures;
  for (auto [i, capture] : llvm::enumerate(op.getExplicitCaptures())) {
    auto noUseInRegion = [i = i](Region *region) {
      return region->getArgument(i).use_empty();
    };
    if (llvm::all_of(op.getPartitionRegions(), noUseInRegion)) {
      unusedArgs.set(i);
      continue;
    }

    auto [it, inserted] = uniqueCaptures.try_emplace(capture, i);
    if (!inserted) {
      unsigned duplicateIdx = it->second;
      b.modifyOpInPlace(op, [&, i = i] {
        for (Region *region : op.getPartitionRegions()) {
          b.replaceAllUsesWith(region->getArgument(i),
                               region->getArgument(duplicateIdx));
        }
      });
      unusedArgs.set(i);
    }
  }
  if (unusedArgs.none() && unusedResults.none())
    return failure();

  if (unusedArgs.any()) {
    b.modifyOpInPlace(op, [&] {
      for (Region *region : op.getPartitionRegions())
        region->front().eraseArguments(unusedArgs);
      op->eraseOperands(unusedArgs);
    });
  }

  if (unusedResults.any()) {
    for (Block &block : op.getDefaultRegion()) {
      if (auto yield = dyn_cast<WarpYieldOp>(block.getTerminator())) {
        b.modifyOpInPlace(yield, [&] { yield->eraseOperands(unusedResults); });
      }
    }

    SmallVector<Type> newTypes;
    for (auto [i, type] : llvm::enumerate(op.getResultTypes())) {
      if (!unusedResults.test(i))
        newTypes.push_back(type);
    }
    OperationState state(op.getLoc(), op->getName(), op.getOperands(), newTypes,
                         op->getAttrs());
    state.addRegion()->takeBody(op.getDefaultRegion());
    state.addRegion()->takeBody(op.getPartitionOpHolder());
    auto newOp = cast<WarpSpecializeOp>(b.create(state));
    unsigned newResultIdx = 0;
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      if (!unusedResults.test(i))
        result.replaceAllUsesWith(newOp.getResult(newResultIdx++));
    }
    assert(newResultIdx == newOp.getNumResults());
    b.eraseOp(op);
  }

  return success();
}

void WarpSpecializeOp::build(OpBuilder &builder, OperationState &state,
                             TypeRange resultTypes,
                             ArrayRef<int32_t> partitionNumWarps,
                             unsigned partitionNumRegions) {
  build(builder, state, resultTypes, /*explicitCaptures=*/ValueRange(),
        partitionNumWarps, {}, {}, {});
  OpBuilder::InsertionGuard guard(builder);
  Block *container = builder.createBlock(state.regions.back().get());
  builder.create<WarpSpecializePartitionsOp>(state.location,
                                             partitionNumRegions);
}

void WarpSpecializeOp::build(OpBuilder &builder, OperationState &state,
                             TypeRange resultTypes, ValueRange explicitCaptures,
                             ArrayRef<int32_t> partitionNumWarps) {
  build(builder, state, resultTypes, explicitCaptures, partitionNumWarps, {},
        {}, {});
}

ParseResult WarpSpecializeOp::parse(OpAsmParser &p, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SMLoc operandLoc = p.getCurrentLocation();
  if (p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseOptionalAttrDictWithKeyword(result.attributes) ||
      p.parseKeyword("default") || p.parseRegion(*result.addRegion()))
    return failure();

  OperationState partitionOpState(
      p.getEncodedSourceLoc(p.getCurrentLocation()),
      WarpSpecializePartitionsOp::getOperationName());

  SmallVector<int32_t> partitionNumWarps;
  SmallVector<OpAsmParser::Argument> partitionArgs;
  while (succeeded(p.parseOptionalKeyword(
      ("partition" + Twine(partitionNumWarps.size()).str())))) {
    partitionArgs.clear();
    SMLoc regionLoc = p.getCurrentLocation();
    if (p.parseArgumentList(partitionArgs, AsmParser::Delimiter::Paren,
                            /*allowType=*/true) ||
        p.parseKeyword("num_warps") || p.parseLParen() ||
        p.parseInteger(partitionNumWarps.emplace_back()) || p.parseRParen() ||
        p.parseRegion(*partitionOpState.addRegion(), partitionArgs))
      return failure();
  }

  FunctionType types;
  if (p.parseColon() || p.parseType(types) ||
      p.resolveOperands(operands, types.getInputs(), operandLoc,
                        result.operands))
    return failure();

  result.addTypes(types.getResults());
  result.addAttribute(getPartitionNumWarpsAttrName(result.name),
                      p.getBuilder().getDenseI32ArrayAttr(partitionNumWarps));

  Block &holder = result.addRegion()->emplaceBlock();
  OpBuilder b(p.getContext());
  b.setInsertionPointToStart(&holder);
  b.create(partitionOpState);
  return success();
}

void WarpSpecializeOp::print(OpAsmPrinter &p) {
  p << '(';
  p.printOperands(getOperands());
  p << ')';
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     {getPartitionNumWarpsAttrName()});

  p.printNewline();
  p << "default ";
  p.printRegion(getDefaultRegion(), /*printEntryBlockArgs=*/false);

  for (auto [i, region, numWarps] :
       llvm::enumerate(getPartitionRegions(), getPartitionNumWarps())) {
    p.printNewline();
    p << "partition" << i << '(';
    llvm::interleaveComma(region->getArguments(), p, [&](BlockArgument arg) {
      p.printRegionArgument(arg);
    });
    p << ") num_warps(" << numWarps << ") ";
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
  p << " : ";
  p.printFunctionalType(*this);
}

LogicalResult WarpYieldOp::verify() {
  if (getNumOperands() != getParentOp().getNumResults()) {
    return emitOpError("has ")
           << getNumOperands() << " operands but parent op expected "
           << getParentOp().getNumResults();
  }
  for (auto [i, result, type] :
       llvm::enumerate(getParentOp().getResultTypes(), getOperandTypes())) {
    if (result != type) {
      return emitOpError("operand #") << i << " has type " << type
                                      << " but parent op expected " << result;
    }
  }
  return success();
}

// Get the size of a scalar type when stored in shared memory.
// TODO: Generalize this as needed.
static size_t getSharedMemorySize(Type type) {
  if (isa<IntegerType, FloatType>(type))
    return llvm::divideCeil(type.getIntOrFloatBitWidth(), 8);
  if (isa<PointerType, TensorDescType>(type))
    return 8;
  if (auto desc = dyn_cast<MemDescType>(type)) {
    if (!isa<SharedMemorySpaceAttr>(desc.getMemorySpace()))
      return 8;
    return 8 + desc.getRank() * 4;
  }
  llvm::report_fatal_error(
      Twine("shared memory size for scalar type is unspecified: ") +
      mlir::debugString(type));
}

std::pair<uint64_t, uint64_t> WarpSpecializeOp::getCaptureSizeAlign() {
  uint64_t captureSize = 0;
  // Tightly pack the captures in memory.
  for (Type type : getOperandTypes()) {
    captureSize += getSharedMemorySize(type);
  }
  // Align the captures to 8 bytes.
  return {captureSize, 8};
}

unsigned WarpSpecializeOp::getTotalPartitionWarps() {
  ArrayRef<int32_t> numWarps = getPartitionNumWarps();
  return std::accumulate(numWarps.begin(), numWarps.end(), 0);
}

} // namespace mlir::triton::gpu
