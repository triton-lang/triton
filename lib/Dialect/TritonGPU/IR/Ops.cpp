#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

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
    auto srcType = convert.getSrc().getType();
    auto dstType = convert.getType();
    auto srcLL = toLinearLayout(srcType.getShape(), srcType.getEncoding());
    auto dstLL = toLinearLayout(dstType.getShape(), dstType.getEncoding());
    if (srcLL && dstLL && *srcLL == *dstLL) {
      rewriter.replaceOpWithNewOp<triton::ReshapeOp>(
          op, op.getType(), convert.getSrc(), op.getAllowReorder());
      return mlir::success();
    }
    if (isExpensiveView(convert.getSrc().getType(), op.getType()))
      return failure();
    if (!op.getAllowReorder() || op.getEfficientLayout())
      return failure();

    rewriter.replaceOpWithNewOp<triton::ReshapeOp>(
        op, op.getType(), convert.getSrc(), op.getAllowReorder());
    return mlir::success();
  }
};

// histogram(cvt) -> histogram
struct CanonicalizeConvertFromHistogram
    : public mlir::OpRewritePattern<triton::HistogramOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::HistogramOp op,
                  PatternRewriter &rewriter) const override {
    auto convert = op.getSrc().getDefiningOp<ConvertLayoutOp>();
    if (!convert)
      return failure();
    rewriter.replaceOpWithNewOp<triton::HistogramOp>(
        op, op->getResult(0).getType(), convert.getSrc());
    return mlir::success();
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

    // for hopper MMAv3
    if (mlir::isa<SharedEncodingAttr>(dstType.getEncoding()) &&
        mlir::isa<NvidiaMmaEncodingAttr>(srcType.getEncoding()) &&
        llvm::any_of(op.getResult().getUsers(), [](Operation *dot) {
          return dot->hasTrait<OpTrait::DotLike>();
        })) {
      return failure();
    }

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
                                               histogram.getSrc());
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
                                               sharedLoad.getSrc());

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

  if (!resTy.getElementType().isBF16())
    return emitError() << "only bf16 is supported for now, got "
                       << resTy.getElementType();

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
                      TypedValue<RankedTensorType> src, Type resultElemType,
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

  auto resultTy = RankedTensorType::get(shape, resultElemType, outEnc);
  build(builder, state, resultTy, src, builder.getI32IntegerAttr(axis));
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

LogicalResult MemDescTransOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the input
  auto argTy = cast<MemDescType>(operands[0].getType());
  auto order = properties.as<Properties *>()->order.asArrayRef();
  SmallVector<int64_t> retShape = applyPermutation(argTy.getShape(), order);

  auto retEltTy = argTy.getElementType();
  Attribute argEncoding = argTy.getEncoding();
  Attribute retEncoding;
  if (argEncoding) {
    Dialect &dialect = argEncoding.getDialect();
    auto inferLayoutInterface = cast<DialectInferLayoutInterface>(&dialect);
    if (inferLayoutInterface
            ->inferTransOpEncoding(argEncoding, order, retEncoding)
            .failed()) {
      return failure();
    }
  }
  auto memDescTy = cast<MemDescType>(argTy);
  inferredReturnTypes.push_back(MemDescType::get(
      retShape, retEltTy, retEncoding, memDescTy.getMemorySpace(),
      memDescTy.getMutableMemory()));
  return success();
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
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       mlir::triton::gpu::SharedMemory::get());
  if (getSrc())
    effects.emplace_back(MemoryEffects::Write::get(),
                         getOperation()->getOpResult(0),
                         mlir::triton::gpu::SharedMemory::get());
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

LogicalResult LocalAllocOp::verify() {
  if (!getSrc()) {
    if (!getType().getMutableMemory())
      return emitError("uninitialized alloc must have a mutable memdesc type");
    return success();
  }
  auto srcTy = getSrc().getType();
  auto dstTy = getType();

  if (srcTy.getElementType() != dstTy.getElementType()) {
    return emitError("result element type must match desc element type");
  }
  return success();
}

// LocalLoadOp
void LocalLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// LocalStoreOp
LogicalResult LocalStoreOp::verify() {
  if (!getDst().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return success();
}

void LocalStoreOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(), &getDstMutable(),
                       mlir::triton::gpu::SharedMemory::get());
}

// AsyncCopyGlobalToLocalOp
LogicalResult AsyncCopyGlobalToLocalOp::verify() {
  if (!getResult().getType().getMutableMemory())
    return emitOpError("Cannot store into immutable memory");
  return success();
}

void AsyncCopyGlobalToLocalOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       mlir::triton::GlobalMemory::get());
  effects.emplace_back(MemoryEffects::Write::get(), &getResultMutable(),
                       mlir::triton::gpu::SharedMemory::get());
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

  if (!isa<SharedEncodingAttr>(srcEnc)) {
    return emitError("src encoding must be SharedEncodingAttr");
  }
  if (!isa<SharedEncodingAttr>(dstEnc)) {
    return emitError("result encoding must be SharedEncodingAttr");
  }

  // TODO(jlebar): Currently we generate illegal encodings, so we can't add a
  // verifier for them.  In particular, we use the same encoding for the src and
  // dst of a subview op, when the subview removes a dimension.  That generates
  // an illegal shared encoding (because the size of `order` doesn't match the
  // rank of the tensor), but it's not checked anywhere, and we believe the
  // resulting code ultimately works.

  return success();
}

// -- LocalAllocOp --

int32_t LocalAllocOp::getAlignmentOrDefault() {
  auto align = getAlignment();
  if (align) {
    return *align;
  }

  auto ty = getType();
  auto shapePerCTA = triton::gpu::getShapePerCTA(ty);
  auto bytes =
      product<int64_t>(shapePerCTA) * (ty.getElementTypeBitWidth() / 8);

  // XXX(Keren): magic numbers 256 and 1024
  // Software swizzling calculates phase based on offset, while hardware
  // swizzling do that based on physical address. Thus only by setting the
  // alignment to 1024 can ensure the correctness.
  return bytes > 256 ? 1024 : 8;
}

} // namespace mlir::triton::gpu
