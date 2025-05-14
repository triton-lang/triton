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
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/LogicalResult.h"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include <iterator>
#include <memory>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONREWRITETENSORDESCRIPTORTOPOINTER
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

namespace {

bool hasATensorDescriptorType(mlir::TypeRange types) {
  return llvm::any_of(types, [](mlir::Type t) {
    return llvm::isa<mlir::triton::TensorDescType>(t);
  });
}

/**
 * @brief Convert integer types to signless. Other types are returned as is.
 */
mlir::Type toSignlessIntegerType(mlir::Type t) {
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(t)) {
    return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return t;
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

// Note this has been adapted from RewriteTensorPointer.cpp
Value getExpandedOffsetWithRange(OpBuilder &builder, const Location &loc,
                                 ArrayRef<std::int64_t> blockShape,
                                 ValueRange offsets, unsigned i) {
  // Add range
  auto indexI32RowType =
      RankedTensorType::get({blockShape[i]}, builder.getI32Type());
  auto indexRowType =
      RankedTensorType::get({blockShape[i]}, builder.getI64Type());
  Value splatOffset =
      builder.create<triton::SplatOp>(loc, indexRowType, offsets[i]);
  Value range = builder.create<triton::MakeRangeOp>(loc, indexI32RowType, 0,
                                                    blockShape[i]);
  Value i64Range = builder.create<arith::ExtSIOp>(loc, indexRowType, range);

  // Expand dimensions
  Value expandedResult =
      builder.create<arith::AddIOp>(loc, splatOffset, i64Range);
  for (size_t j = 0; j < blockShape.size(); ++j) {
    if (j == i) {
      continue;
    }
    expandedResult =
        builder.create<triton::ExpandDimsOp>(loc, expandedResult, j);
  }

  return expandedResult;
}

// Note this has been adapted from RewriteTensorPointer.cpp
Value generatePtr(OpBuilder &builder, const Location &loc,
                  ArrayRef<std::int64_t> blockShape, Value base,
                  ValueRange strides, ValueRange offsets) {
  assert(blockShape.size() == offsets.size() &&
         blockShape.size() == strides.size());
  auto indexTensorType =
      RankedTensorType::get(blockShape, builder.getI64Type());
  auto ptrType = cast<triton::PointerType>(base.getType());
  auto ptrTensorType = RankedTensorType::get(blockShape, ptrType);

  // Generate offsets per dimension
  Value ptr = builder.create<triton::SplatOp>(loc, ptrTensorType, base);
  for (unsigned i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets, i);

    // We must splat strides into the expanded shape not a row for retaining
    // the divisibility information given by strides
    Value splatStride = builder.create<triton::SplatOp>(
        loc, offsetWithRange.getType(), strides[i]);
    Value offsetWithStride =
        builder.create<arith::MulIOp>(loc, offsetWithRange, splatStride);
    Value broadcasted = builder.create<triton::BroadcastOp>(
        loc, indexTensorType, offsetWithStride);

    // Add to the pointer
    ptr =
        builder.create<triton::AddPtrOp>(loc, ptrTensorType, ptr, broadcasted);
  }

  return ptr;
}

// Note this has been adapted from RewriteTensorPointer.cpp
Value generateMask(OpBuilder &builder, const Location &loc,
                   ArrayRef<std::int64_t> blockShape, ValueRange offsets,
                   ValueRange shape) {
  assert(blockShape.size() == shape.size() &&
         blockShape.size() == offsets.size());

  // Generate mask per dimension
  auto maskTensorType = RankedTensorType::get(blockShape, builder.getI1Type());
  Value mask;
  for (std::size_t i = 0; i < blockShape.size(); ++i) {
    auto offsetWithRange =
        getExpandedOffsetWithRange(builder, loc, blockShape, offsets, i);

    // Compare with lower bound
    Value lowerBound = builder.create<mlir::arith::ConstantIntOp>(
        loc, 0, builder.getI64Type());
    Value splatLowerBound = builder.create<triton::SplatOp>(
        loc, offsetWithRange.getType(), lowerBound);
    Value cmpLower = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, offsetWithRange, splatLowerBound);

    // Compare with upper bound
    Value splatUpperBound = builder.create<triton::SplatOp>(
        loc, offsetWithRange.getType(), shape[i]);
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

// Note this has been adapted from RewriteTensorPointer.cpp. It appears
// to be getting the values used for the masked out elements
Value generateOther(OpBuilder &builder, const Location &loc, Value base,
                    ArrayRef<std::int64_t> blockShape) {
  // Create element attribute
  auto elementType = cast<triton::PointerType>(base.getType()).getPointeeType();
  auto otherTensorType = RankedTensorType::get(blockShape, elementType);

  // Set zero padding value (the default)
  TypedAttr attr = builder.getZeroAttr(elementType);

  // Create tensor
  Value constant = builder.create<arith::ConstantOp>(loc, attr);
  return builder.create<triton::SplatOp>(loc, otherTensorType, constant);
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
  matchAndRewrite(triton::MakeTensorDescOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<mlir::Value> ptrShapeStrides;
    // Note that none of these values come from a tensor descriptor so its safe
    // to get these directly from the op
    llvm::append_values(ptrShapeStrides, op.getBase());
    llvm::append_range(ptrShapeStrides, castToI64(rewriter, op.getShape()));
    llvm::append_range(ptrShapeStrides, op.getStrides());
    rewriter.replaceOpWithMultiple(op, {ptrShapeStrides});
    return mlir::success();
  }
};

struct RewriteLoadPattern : OpConversionPattern<triton::DescriptorLoadOp> {
  using OpConversionPattern<triton::DescriptorLoadOp>::OpConversionPattern;

  llvm::LogicalResult
  matchAndRewrite(triton::DescriptorLoadOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    const auto rank = blockShape.size();
    assert(adaptor.getDesc().size() == 1 + 2 * rank &&
           "Expected tensor descriptors to be "
           "broken down into a ptr and "
           "`rank` shapes and `rank` strides");

    auto base = adaptor.getDesc().front();
    auto shape = adaptor.getDesc().slice(1, rank);
    auto strides = adaptor.getDesc().slice(1 + rank, rank);
    // Note that indices aren't converted so
    // we can get them directly here
    auto offsets = castToI64(rewriter, op.getIndices());

    auto newLoad = rewriter.replaceOpWithNewOp<triton::LoadOp>(
        op,
        generatePtr(rewriter, op->getLoc(), blockShape, base, strides, offsets),
        generateMask(rewriter, op->getLoc(), blockShape, offsets, shape),
        generateOther(rewriter, op->getLoc(), base, blockShape),
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
    const auto blockShape = op.getDesc().getType().getBlockType().getShape();
    const auto rank = blockShape.size();
    assert(adaptor.getDesc().size() == 1 + 2 * rank &&
           "Expected tensor descriptors to be "
           "broken down into a ptr and "
           "`rank` shapes and `rank` strides");

    auto base = adaptor.getDesc().front();
    auto shape = adaptor.getDesc().slice(1, rank);
    auto strides = adaptor.getDesc().slice(1 + rank, rank);
    // Note that indices aren't converted so
    // we can get them directly here
    auto offsets = castToI64(rewriter, op.getIndices());

    auto newStore = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op,
        generatePtr(rewriter, op->getLoc(), blockShape, base, strides, offsets),
        op.getSrc(),
        generateMask(rewriter, op->getLoc(), blockShape, offsets, shape),
        triton::CacheModifier::NONE, triton::EvictionPolicy::NORMAL);
    newStore->setAttrs(filterSegmentSizes(op->getAttrs()));

    return llvm::success();
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
      // for each dimension, i.e., we create 1+2*rank values. Note that tensor
      // descriptors may be signed/unsigned integers whereas pointers should
      // always be signless.
      auto tensorType = t.getBlockType();
      out.push_back(triton::getPointerType(
          toSignlessIntegerType(tensorType.getElementType())));
      out.insert(out.end(), 2 * tensorType.getRank(),
                 mlir::IntegerType::get(t.getContext(), 64));
      return mlir::success();
    });

    mlir::RewritePatternSet patterns(op->getContext());

    // Populate conversion patterns to handle loops, function calls, and arith
    // ops.
    triton::populateFunctionTypeConversions(converter, patterns);
    mlir::scf::populateSCFStructuralTypeConversions(converter, patterns);
    triton::populateArithTypeConversions(converter, patterns);

    patterns
        .add<RewriteMakeTensorDesc, RewriteLoadPattern, RewriteStorePattern>(
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
