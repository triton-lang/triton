#include "triton/Dialect/Triton/Transforms/ArithTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/FunctionTypeConversion.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <iterator>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONREWRITETENSORDESCRIPTORTOPOINTER
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

using namespace mlir;

/**
 * @brief Filter out operand segment sizes from the list of attributes since
 * this attribute is operation specific and shouldn't be set arbitrarily.
 */
mlir::SmallVector<NamedAttribute>
filterSegmentSizes(mlir::ArrayRef<NamedAttribute> attrs) {
  mlir::SmallVector<NamedAttribute> ret;
  llvm::copy_if(attrs, std::back_inserter(ret), [](const NamedAttribute &attr) {
    auto attrName = attr.getName().getValue();
    return attrName != "operandSegmentSizes";
  });
  return ret;
}

struct Descriptor {
  Value base;
  ValueRange shape;
  ValueRange strides;
  Value paddingOption;
};

Descriptor unpackDescriptor(TensorDescType type, ValueRange pack) {
  int rank = type.getBlockType().getRank();
  assert(pack.size() == 1 + 2 * static_cast<size_t>(rank) + 1 &&
         "Expected tensor descriptors to consist of a pointer, "
         "followed by 'rank' shape values and 'rank' stride values, "
         "followed by a padding option value.");

  Descriptor res;
  res.base = pack[0];
  res.shape = pack.slice(1, rank);
  res.strides = pack.slice(1 + rank, rank);
  res.paddingOption = pack[1 + 2 * rank];
  return res;
}

Value expandOffsets(OpBuilder &builder, Location loc,
                    ArrayRef<int64_t> blockShape, Value offsets, unsigned dim) {
  Value expandedResult = offsets;
  for (size_t j = 0; j < blockShape.size(); ++j) {
    if (j == dim) {
      continue;
    }
    expandedResult =
        builder.create<triton::ExpandDimsOp>(loc, expandedResult, j);
  }

  return expandedResult;
}

Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                 ArrayRef<std::int64_t> blockShape,
                                 Value offset, unsigned dim) {
  // Add range
  auto indexI32RowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI32Type());
  auto indexRowType =
      RankedTensorType::get({blockShape[dim]}, builder.getI64Type());
  Value splatOffset =
      builder.create<triton::SplatOp>(loc, indexRowType, offset);
  Value range = builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0,
                                                    blockShape[dim]);
  Value i64Range = builder.create<arith::ExtSIOp>(loc, indexRowType, range);

  Value offsets = builder.create<arith::AddIOp>(loc, splatOffset, i64Range);
  return expandOffsets(builder, loc, blockShape, offsets, dim);
}

Value generatePtrFromOffsetRanges(OpBuilder &builder, Location loc,
                                  ArrayRef<int64_t> blockShape,
                                  Descriptor &desc, ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  auto indexTensorType =
      RankedTensorType::get(blockShape, builder.getI64Type());
  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);

  // Generate offsets per dimension
  Value ptr = builder.create<triton::SplatOp>(loc, ptrTensorType, desc.base);
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    // We must splat strides into the expanded shape not a row for retaining
    // the divisibility information given by strides
    Value splatStride = builder.create<triton::SplatOp>(
        loc, offsets[i].getType(), desc.strides[i]);
    Value offsetWithStride =
        builder.create<arith::MulIOp>(loc, offsets[i], splatStride);
    Value broadcasted = builder.create<triton::BroadcastOp>(
        loc, indexTensorType, offsetWithStride);

    // Add to the pointer
    ptr =
        builder.create<triton::AddPtrOp>(loc, ptrTensorType, ptr, broadcasted);
  }

  return ptr;
}

Value generatePtr(OpBuilder &builder, const Location &loc,
                  ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                  ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                     offsetRanges);
}

Value generateMaskFromOffsetRanges(OpBuilder &builder, const Location &loc,
                                   ArrayRef<std::int64_t> blockShape,
                                   Descriptor &desc, ValueRange offsetRanges) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsetRanges.size());

  // Generate mask per dimension
  auto maskTensorType = RankedTensorType::get(blockShape, builder.getI1Type());
  Value mask;
  for (std::size_t i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange = offsetRanges[i];

    // Compare with lower bound
    Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
        loc, builder.getI64Type(), 0);
    Value splatLowerBound = builder.create<triton::SplatOp>(
        loc, offsetWithRange.getType(), lowerBound);
    Value cmpLower = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, offsetWithRange, splatLowerBound);

    // Compare with upper bound
    Value splatUpperBound = builder.create<triton::SplatOp>(
        loc, offsetWithRange.getType(), desc.shape[i]);
    Value cmpUpper = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, offsetWithRange, splatUpperBound);

    // And and broadcast
    Value andResult = builder.create<arith::AndIOp>(loc, cmpLower, cmpUpper);
    Value broadcasted =
        builder.create<triton::BroadcastOp>(loc, maskTensorType, andResult);

    // And up all results
    if (!mask) {
      mask = broadcasted;
    } else {
      mask = builder.create<arith::AndIOp>(loc, mask, broadcasted);
    }
  }

  return mask;
}

Value generateMask(OpBuilder &builder, const Location &loc,
                   ArrayRef<std::int64_t> blockShape, Descriptor &desc,
                   ValueRange offsets) {
  assert(blockShape.size() == desc.shape.size());
  assert(blockShape.size() == offsets.size());
  SmallVector<Value> offsetRanges;
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets[i], i);
    offsetRanges.push_back(offsetWithRange);
  }

  return generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                      offsetRanges);
}

Value generateOther(OpBuilder &builder, Location loc, Type scalarTy,
                    ArrayRef<int64_t> blockShape,
                    Value paddingOption = nullptr) {
  auto blockTy = RankedTensorType::get(blockShape, scalarTy);
  if (paddingOption && mlir::isa<FloatType>(scalarTy)) {
    auto floatTy = mlir::cast<FloatType>(scalarTy);
    auto nan = llvm::APFloat::getNaN(floatTy.getFloatSemantics());
    auto nanValue = builder.create<arith::ConstantOp>(
        loc,
        SplatElementsAttr::get(blockTy, builder.getFloatAttr(floatTy, nan)));
    auto zeroValue = builder.create<arith::ConstantOp>(
        loc, SplatElementsAttr::get(blockTy, builder.getZeroAttr(floatTy)));
    return builder.create<mlir::arith::SelectOp>(loc, paddingOption, nanValue,
                                                 zeroValue);
  } else {
    auto attr = builder.getZeroAttr(blockTy);
    return builder.create<arith::ConstantOp>(loc, attr);
  }
}

Value generateOther(OpBuilder &builder, Location loc, TensorDescType descTy,
                    Value paddingOption = nullptr) {
  auto blockTy = descTy.getSignlessBlockType();
  return generateOther(builder, loc, blockTy.getElementType(),
                       blockTy.getShape(), paddingOption);
}

SmallVector<mlir::Value> castToI64(OpBuilder &builder,
                                   mlir::ValueRange values) {
  auto i64Type = builder.getI64Type();
  return llvm::map_to_vector(values, [&](mlir::Value v) {
    return builder.createOrFold<arith::ExtSIOp>(v.getLoc(), i64Type, v);
  });
}

struct RewriteMakeTensorDesc : OpConversionPattern<triton::MakeTensorDescOp> {
  using OpConversionPattern<triton::MakeTensorDescOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::MakeTensorDescOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Value> ptrShapeStridesPaddingOption;
    llvm::append_values(ptrShapeStridesPaddingOption, adaptor.getBase());
    llvm::append_range(ptrShapeStridesPaddingOption,
                       castToI64(rewriter, adaptor.getShape()));
    llvm::append_range(ptrShapeStridesPaddingOption, adaptor.getStrides());
    auto paddingOption = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(adaptor.getPadding() ==
                             triton::PaddingOption::PAD_NAN));
    llvm::append_values(ptrShapeStridesPaddingOption, paddingOption);
    rewriter.replaceOpWithMultiple(op, {ptrShapeStridesPaddingOption});
    return mlir::success();
  }
};

struct RewriteLoadPattern : OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    auto descTy = op.getDesc().getType();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());
    auto other = generateOther(rewriter, loc, descTy, desc.paddingOption);
    auto newLoad = rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, generatePtr(rewriter, loc, blockShape, desc, offsets),
        generateMask(rewriter, loc, blockShape, desc, offsets), other,
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

struct RewriteStorePattern : OpConversionPattern<triton::DescriptorStoreOp> {
  using OpConversionPattern<triton::DescriptorStoreOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorStoreOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getBlockType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());

    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
        generateMask(rewriter, loc, blockShape, desc, offsets),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

std::pair<Value, Value>
generateGatherScatterPtrMask(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> blockShape, Descriptor &desc,
                             Value xOffsets, Value yOffset) {
  Value xOffsetRange =
      expandOffsets(builder, loc, blockShape, xOffsets, /*dim=*/0);
  yOffset = castToI64(builder, {yOffset})[0];
  auto xOffsetI64Ty = RankedTensorType::get(
      cast<RankedTensorType>(xOffsetRange.getType()).getShape(),
      yOffset.getType());
  xOffsetRange =
      builder.create<arith::ExtSIOp>(loc, xOffsetI64Ty, xOffsetRange);
  auto yOffsetRange =
      getExpandedOffsetWithRange(builder, loc, blockShape, yOffset, /*dim=*/1);
  auto ptr = generatePtrFromOffsetRanges(builder, loc, blockShape, desc,
                                         {xOffsetRange, yOffsetRange});
  auto mask = generateMaskFromOffsetRanges(builder, loc, blockShape, desc,
                                           {xOffsetRange, yOffsetRange});
  return {ptr, mask};
}

struct RewriteGatherPattern : OpConversionPattern<triton::DescriptorGatherOp> {
  using OpConversionPattern<triton::DescriptorGatherOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorGatherOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = op.getResult().getType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto [ptr, mask] = generateGatherScatterPtrMask(
        rewriter, loc, blockShape, desc, op.getXOffsets(), op.getYOffset());
    auto other = generateOther(rewriter, loc,
                               descTy.getSignlessBlockType().getElementType(),
                               blockShape, desc.paddingOption);
    auto newLoad = rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op, ptr, mask, other, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, false);
    newLoad->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

struct RewriteScatterPattern
    : OpConversionPattern<triton::DescriptorScatterOp> {
  using OpConversionPattern<triton::DescriptorScatterOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorScatterOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = op.getSrc().getType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto [ptr, mask] = generateGatherScatterPtrMask(
        rewriter, loc, blockShape, desc, op.getXOffsets(), op.getYOffset());
    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, ptr, op.getSrc(), mask, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
  }
};

std::optional<RMWOp> translateReduceKind(DescriptorReduceKind kind,
                                         TensorDescType ty) {
  auto scalarTy = ty.getBlockType().getElementType();
  switch (kind) {
  case DescriptorReduceKind::ADD:
    return scalarTy.isInteger() ? RMWOp::ADD : RMWOp::FADD;
  case DescriptorReduceKind::MIN:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMIN;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MIN;
    }
    return {};
  case DescriptorReduceKind::MAX:
    if (scalarTy.isUnsignedInteger()) {
      return RMWOp::UMAX;
    } else if (scalarTy.isSignedInteger()) {
      return RMWOp::MAX;
    }
    return {};
  case DescriptorReduceKind::AND:
    return RMWOp::AND;
  case DescriptorReduceKind::OR:
    return RMWOp::OR;
  case DescriptorReduceKind::XOR:
    return RMWOp::XOR;
  default:
    break;
  }
  return {};
}

struct RewriteReducePattern : OpConversionPattern<triton::DescriptorReduceOp> {
  using OpConversionPattern<triton::DescriptorReduceOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorReduceOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto descTy = op.getDesc().getType();
    const auto blockShape = descTy.getBlockType().getShape();
    auto desc = unpackDescriptor(descTy, adaptor.getDesc());
    auto offsets = castToI64(rewriter, op.getIndices());
    auto rmwOp = translateReduceKind(op.getKind(), descTy);
    if (!rmwOp) {
      std::string msgstring;
      llvm::raw_string_ostream msg(msgstring);
      msg << "Cannot fallback on descriptor atomic op, unsupported for type "
          << descTy.getBlockType().getElementType();
      return op->emitError(msgstring);
    }

    rewriter.create<triton::AtomicRMWOp>(
        loc, descTy.getSignlessBlockType(), *rmwOp,
        generatePtr(rewriter, loc, blockShape, desc, offsets), op.getSrc(),
        generateMask(rewriter, loc, blockShape, desc, offsets),
        MemSemantic::RELEASE, MemSyncScope::GPU);
    op.erase();
    return success();
  }
};

/**
 * @brief This implements the pass for converting triton tensor descriptor
 * loads/stores into indexed loads/stores.
 *
 * The key idea is that each tensor descriptor can be broken down into multiple
 * values. Suppose we have a tensor pointer with rank r, we can cast that tensor
 * descriptor value to and from 1+2r values: a tensor pointer value and two i32
 * value for each dimension representing the dynamic shape and strides.
 *
 * As in normal conversion patterns, individual operations can be converted
 * using casted tensor descriptors and offsets and casting the results back to
 * tensor pointers.
 *
 * We have special handling for TMA loads/stores and the make tensor descriptor
 * op.
 *
 * @note Why use the conversion pattern rewriter? In most cases the defining
 * operation of a tensor descriptor will be a make tensor descriptor op.
 * However, this isn't always true - for example, if the tensor descriptor is a
 * function argument or is in a conditional statement, we need better tracking
 * of the pointer, shape, and strides.
 */
class TritonRewriteTensorDescriptorToPointerPass
    : public impl::TritonRewriteTensorDescriptorToPointerBase<
          TritonRewriteTensorDescriptorToPointerPass> {
  void runOnOperation() override {
    auto op = getOperation();

    mlir::ConversionTarget target(getContext());
    target.addDynamicallyLegalDialect<mlir::arith::ArithDialect,
                                      mlir::scf::SCFDialect,
                                      mlir::triton::TritonDialect>(
        [](mlir::Operation *op) {
          return !hasATensorDescriptorType(op->getOperandTypes()) &&
                 !hasATensorDescriptorType(op->getResultTypes());
        });
    target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
      return !hasATensorDescriptorType(funcOp.getFunctionType().getInputs()) &&
             !hasATensorDescriptorType(funcOp.getFunctionType().getResults());
    });

    mlir::TypeConverter converter;

    converter.addConversion([](mlir::Type t) {
      // Most types don't require any conversion
      return t;
    });
    converter.addConversion([](mlir::triton::TensorDescType t,
                               llvm::SmallVectorImpl<mlir::Type> &out) {
      // We convert a tensor descriptor into an pointer, and a shape and stride
      // for each dimension, and padding option. i.e., we create 1+2*rank+1
      // values. Note that tensor descriptors may be signed/unsigned integers
      // whereas pointers should always be signless.
      auto tensorType = t.getSignlessBlockType();
      out.push_back(triton::getPointerType(tensorType.getElementType()));
      out.insert(out.end(), 2 * tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 64));
      out.push_back(mlir::IntegerType::get(t.getContext(), 1));
      return mlir::success();
    });

    mlir::RewritePatternSet patterns(op->getContext());

    // Populate conversion patterns to handle loops, function calls, and arith
    // ops.
    triton::populateFunctionTypeConversions(converter, patterns);
    mlir::scf::populateSCFStructuralTypeConversions(converter, patterns);
    triton::populateArithTypeConversions(converter, patterns);

    patterns
        .add<RewriteMakeTensorDesc, RewriteLoadPattern, RewriteStorePattern,
             RewriteGatherPattern, RewriteScatterPattern, RewriteReducePattern>(
            converter, &getContext());

    ConversionConfig config;
    config.buildMaterializations = false;

    if (mlir::failed(mlir::applyPartialConversion(
            op, target, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::triton
