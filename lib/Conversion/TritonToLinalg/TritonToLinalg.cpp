//===----------------------------------------------------------------------===//
//
// Copyright (c) Triton Project Contributors.
//
//===----------------------------------------------------------------------===//

#include "triton/Conversion/TritonToLinalg/TritonToLinalg.h"
#include "triton/Analysis/MaskAnalysis.h"
#include "triton/Analysis/PtrAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

#include <numeric>

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton/Conversion/TritonToLinalg/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat and sitofp) that produce it and to extract they
// underlying scalar value . If no scalar value can be extracted, a nullptr is
// returned.
static std::optional<Value>
getScalarValue(Value v, Location loc, ConversionPatternRewriter &rewriter) {
  // Record if an sitofp op was in the chain of ops that produce the scalar
  Operation *siToFp = nullptr;

  while (true) {
    if (!v.getType().dyn_cast<ShapedType>()) {
      break;
    } else if (auto op = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = op.getValue().dyn_cast<DenseElementsAttr>()) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp =
            rewriter.create<arith::ConstantOp>(op.getLoc(), elemValue);
        v = constOp.getResult();
      }
    } else if (auto op = v.getDefiningOp<triton::SplatOp>()) {
      v = op.getSrc();
    } else if (auto op = v.getDefiningOp<arith::SIToFPOp>()) {
      siToFp = op;
      v = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }

  if (siToFp) {
    auto resType = siToFp->getResult(0).getType();
    if (auto shapedType = dyn_cast<ShapedType>(resType)) {
      resType = shapedType.getElementType();
    }
    return rewriter.create<arith::SIToFPOp>(loc, resType, v);
  }
  return v;
}

static SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

static Value getTransposedValue(Value source, const Location loc,
                                ConversionPatternRewriter &rewriter) {

  auto sourceType = source.getType().cast<RankedTensorType>();
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(sourceRank);
  std::iota(std::begin(perm), std::end(perm), 0);
  std::swap(perm[sourceRank - 1], perm[sourceRank - 2]);

  SmallVector<int64_t> transposedShape(sourceType.getShape());
  std::swap(transposedShape[sourceRank - 1], transposedShape[sourceRank - 2]);

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = op.getType().cast<TensorType>();
    auto loc = op.getLoc();

    auto init = rewriter.create<tensor::EmptyOp>(loc, opType.getShape(),
                                                 opType.getElementType());

    auto filledTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{adaptor.getSrc()},
                                    ValueRange{init})
            .result();

    rewriter.replaceOp(op, filledTensor);
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
private:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) const {
    SmallVector<int64_t> broadcastDims;
    auto srcShape = src.getShape();
    auto dstShape = dst.getShape();

    for (size_t i = 0; i < srcShape.size(); i++) {
      if (dstShape[i] != srcShape[i]) {
        assert(srcShape[i] == 1);
        broadcastDims.push_back(i);
      }
    }
    assert(!broadcastDims.empty() && "cannot identify broadcast dimension");
    return broadcastDims;
  }

  // Broadcasts input tensor based on TosaToLinalg's broadcastToShape
  AffineMap getBroadcastAffineMap(MLIRContext *context,
                                  ArrayRef<int64_t> inputShape,
                                  ArrayRef<int64_t> broadcastToShape) const {

    assert(broadcastToShape.size() >= inputShape.size());

    // Create affine map and shapes for tensor initialization.
    SmallVector<AffineExpr> outExpr;

    size_t diff = broadcastToShape.size() - inputShape.size();
    for (size_t i = 0; i < broadcastToShape.size(); i++) {
      if (i < diff) {
        continue;
      }
      size_t j = i - diff;
      if (inputShape[j] == 1) {
        // Broadcast singleton dimension
        outExpr.push_back(mlir::getAffineConstantExpr(0, context));
        continue;
      }
      // Non-broadcast case
      outExpr.push_back(mlir::getAffineDimExpr(i, context));
    }
    return AffineMap::get(broadcastToShape.size(), 0, outExpr, context);
  }

public:
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op->getNumResults() == 1 && "code assumes single result!");
    RankedTensorType sourceType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    RankedTensorType resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + op->getNumResults());

    indexingMaps.push_back(getBroadcastAffineMap(
        op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(),
                        rewriter.getMultiDimIdentityMap(resultRank));

    assert(op->getNumResults() == 1 && "code assumes single result!");
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 elementType);

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), ValueRange{adaptor.getSrc()},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value opResult = blockArgs[0];
          nestedBuilder.create<linalg::YieldOp>(loc, opResult);
        });

    linalgOp->setAttr("broadcastDims",
                      rewriter.getDenseI64ArrayAttr(
                          getBroadcastDims(sourceType, resultType)));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto srcRank = src.getType().cast<RankedTensorType>().getRank();
    auto resType = op->getResultTypes()[0].cast<RankedTensorType>();
    SmallVector<ReassociationIndices> reassoc;
    int64_t c = 0;
    for (int64_t i = 0; i < srcRank; i++) {
      ReassociationIndices g;
      g.push_back(c++);
      if (op.getAxis() == i)
        g.push_back(c++);
      else if (op.getAxis() == i + 1 && i == srcRank - 1)
        g.push_back(c++);
      reassoc.push_back(g);
    }

    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        op.getLoc(), resType, src, reassoc);

    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
  }
};

struct TransposeConverter : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto srcRank = src.getType().cast<ShapedType>().getRank();
    assert(srcRank == 2 && "only expect transposing 2D data");

    auto res = getTransposedValue(src, op.getLoc(), rewriter);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = op.getResult().getType().cast<TensorType>();
    auto shape = type.getShape();
    auto elementType = type.getElementType();
    auto context = rewriter.getContext();

    assert(type.getShape().size() == 1 &&
           type.getElementType().getIntOrFloatBitWidth() == 32 &&
           "make range can only return 1D int32 tensor");

    SmallVector<AffineMap> indexingMaps{AffineMap::get(
        /* dimCount */ 1, /* symbolCount */ 0,
        SmallVector<AffineExpr>{mlir::getAffineDimExpr(0, context)}, context)};

    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), /* operands */ ValueRange{},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value index = nestedBuilder.create<linalg::IndexOp>(loc, 0);
          Value res = nestedBuilder.create<arith::IndexCastOp>(
              loc, type.getElementType(), index);
          nestedBuilder.create<linalg::YieldOp>(loc, res);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knwonPtrs;
    PtrAnalysis::rewriteAddptrOp(op, rewriter, knwonPtrs);
    return success();
  }
};

struct AssertConverter : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto condVal = op.getCondition();

    if (condVal.getType().isa<mlir::TensorType>()) {
      auto scalarVal = getScalarValue(op.getCondition(), op.getLoc(), rewriter);
      condVal = scalarVal.value_or(condVal);
    }
    assert(condVal && condVal.getType().isa<mlir::IntegerType>() &&
           "Only asserts on scalars are currently supported");

    if (!condVal.getType().isInteger(1)) {
      auto zero =
          rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 32);
      auto newCond = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, condVal, zero);
      condVal = newCond.getResult();
    }

    auto assertMessage =
        llvm::formatv("{0}.py:{1}: {2} Assertion `{3}` failed", op.getFile(),
                      op.getLine(), op.getFunc(), op.getMessage());
    auto assertOp = rewriter.create<mlir::cf::AssertOp>(op.getLoc(), condVal,
                                                        assertMessage.str());

    rewriter.eraseOp(op);
    return success();
  }
};

struct BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto arithBitcast = rewriter.create<arith::BitcastOp>(
        op.getLoc(), op.getType(), op.getOperand());

    rewriter.replaceOp(op, arithBitcast.getResult());
    return success();
  }
};

struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
private:
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto mask = op.getMask();
    auto other = op.getOther();
    auto loc = op.getLoc();

    // 0. Shortcut for scalar loads
    if (!op.getResult().getType().isa<ShapedType>()) {
      // Temporarily disbale scalar load until later passes support it
      op.emitError("Scalar load is currently not supported");
      return failure();

      auto sMemRef = PtrAnalysis::getScalarMemRef(op.getPtr(), adaptor.getPtr(),
                                                  loc, rewriter);
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      auto loadOp = rewriter.create<AffineLoadOp>(op.getLoc(), sMemRef, zeroMap,
                                                  std::nullopt);
      rewriter.replaceOp(op, loadOp.getResult());
      return success();
    }

    // 1. Simple case where no mask is used.
    auto type = ptr.getType().cast<MemRefType>();
    auto tensorType =
        RankedTensorType::get(type.getShape(), type.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));

    if (!mask) {
      assert(!other && "other value used in non-masked load");
      rewriter.create<memref::CopyOp>(loc, ptr, alloc);
      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /* restrict */, true /* writable */);
      rewriter.replaceOp(op, tensor);
      return success();
    }

    // 2. Continuous masked loads.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed())
      return failure();

    auto castOp = ptr.getDefiningOp<memref::ReinterpretCastOp>();
    assert(castOp);
    ptr = castOp.getResult();

    auto srcSubview = mstate.getSubview(ptr, loc, rewriter);
    auto dstSubview = mstate.getSubview(alloc, loc, rewriter);

    // fill load destination with other value
    if (other) {
      auto scalarOther = getScalarValue(other, loc, rewriter);
      assert(scalarOther.has_value() &&
             "other value used in masked load produced by "
             "unsupported instruction");

      // For each dimension check if mstate.dims[i] < shape[i], or-accumulate
      // the result
      auto shape = type.getShape();
      auto accBase =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
              .getResult();
      for (size_t i = 0; i < type.getShape().size(); i++) {
        auto shapei = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(shape[i]));

        Value dimi = mstate.dims[i].dyn_cast<Value>();
        if (!dimi) {
          dimi = rewriter.create<arith::ConstantOp>(
              loc, mstate.dims[i].get<Attribute>().cast<IntegerAttr>());
        }

        auto cmpOp = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, dimi, shapei);
        accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmpOp.getResult())
                      .getResult();
      }

      // condition the memset on the or-accumulation
      // initialize with padding prior to CopyOp
      rewriter.create<scf::IfOp>(
          loc, accBase, [&](OpBuilder &builder, Location loc) {
            builder.create<linalg::FillOp>(loc, ValueRange{scalarOther.value()},
                                           ValueRange{alloc});
            builder.create<scf::YieldOp>(loc);
          });
    }

    rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }
};

struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto val = adaptor.getValue();
    auto mask = op.getMask();
    auto loc = op.getLoc();

    // 0. Shortcut for scalar stores
    if (!val.getType().isa<ShapedType>()) {
      auto sMemRef =
          PtrAnalysis::getScalarMemRef(op.getPtr(), ptr, loc, rewriter);
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      rewriter.create<AffineStoreOp>(loc, val, sMemRef, zeroMap, std::nullopt);
      rewriter.eraseOp(op);
      return success();
    }

    // 1. Simple case where no mask is used.
    if (!mask) {
      rewriter.create<memref::TensorStoreOp>(loc, val, ptr);
      rewriter.eraseOp(op);
      return success();
    }

    // 2. Continuous masked stores.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed())
      return failure();

    auto srcSlice = mstate.getExtractSlice(val, loc, rewriter);
    auto dstSubview = mstate.getSubview(ptr, loc, rewriter);

    rewriter.create<memref::TensorStoreOp>(loc, srcSlice, dstSubview);
    rewriter.eraseOp(op);

    return success();
  }
};

struct LoopConverter : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrAnalysis::IndexMapSet
        levelToBlockArgIndex; // level -> set of block arg index to be replaced

    PtrAnalysis::rewriteForOp(op, rewriter, levelToBlockArgIndex, 0, knownPtrs);
    return success();
  }
};

struct YieldConverter : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstType = op.getType().cast<RankedTensorType>();
    auto loc = op.getLoc();

    auto opa = adaptor.getA();
    auto opb = adaptor.getB();
    auto opc = adaptor.getC();
    auto opcOrig = op.getC();

    bool skipC = false;
    if (auto splatOp = opcOrig.getDefiningOp<triton::SplatOp>()) {
      if (auto val = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
        if (val.getValue().cast<FloatAttr>().getValueAsDouble() == 0.) {
          skipC = true;
        }
      }
    } else if (auto constOp = opcOrig.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat() &&
            denseAttr.getSplatValue<FloatAttr>().getValueAsDouble() == 0.) {
          skipC = true;
        }
      }
    }

    auto init = rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(),
                                                 dstType.getElementType());

    auto res = rewriter
                   .create<linalg::MatmulOp>(loc, ValueRange{opa, opb},
                                             ValueRange{init})
                   .getResult(0);

    if (!skipC) {
      res = rewriter.create<arith::AddFOp>(loc, res, opc);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ReduceConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

private:
  llvm::SmallVector<Operation *> getRedOps(triton::ReduceOp redOp) const {
    auto reduceBlock = redOp.getBody();
    llvm::SmallVector<Operation *> ops;
    for (auto &op : reduceBlock->without_terminator()) {
      ops.push_back(&op);
    }
    return ops;
  }

  bool isReductionOpSupported(Operation *redOp) const {
    return isa<arith::AddFOp, arith::MaxFOp>(redOp);
  }

  float getRedBaseVal(Operation *redOp) const {
    return llvm::TypeSwitch<Operation *, float>(redOp)
        .Case([](arith::AddFOp) { return 0; })
        .Case([](arith::MaxFOp) {
          return -std::numeric_limits<float>::infinity();
        })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return -1;
        });
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    return elemType.isa<FloatType>() &&
           elemType.getIntOrFloatBitWidth() <
               Float32Type::get(elemType.getContext()).getWidth() &&
           isa<arith::AddFOp>(redOp);
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .Case([&](arith::AddFOp) {
          if (convertLhsToF32Precision) {
            lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                          lhs);
          }
          return b.create<arith::AddFOp>(loc, lhs, rhs);
        })
        .Case([&](arith::MaxFOp) {
          return b.create<arith::MaxFOp>(loc, lhs, rhs);
        })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  LogicalResult
  convertToLinalgReduce(triton::ReduceOp op,
                        typename triton::ReduceOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto source = adaptor.getOperands().front();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto elemType = sourceType.getElementType();
    auto resType = op.getResult().front().getType();
    auto loc = op.getLoc();
    auto reductionOps = getRedOps(op);

    // Reduction of arbitrary operations isn't supported because using the first
    // element across the reduction dimension requires us to iterate over a
    // subview that skips over each first element.
    if (reductionOps.size() != 1 ||
        !isReductionOpSupported(reductionOps.front())) {
      return op.emitError("Only support lowering reduction with body "
                          "containing 1 maxf or addf.");
    }

    auto rop = reductionOps.front();
    auto axis = op.getAxis();
    auto isVectorReduce = sourceType.getRank() == 1;

    if (axis == sourceType.getRank() - 1 && !isVectorReduce) {
      source = getTransposedValue(source, op.getLoc(), rewriter);
      axis = sourceType.getRank() - 2;
    }

    bool convertToF32Precision = requiresF32Conversion(resType, rop);

    auto constantType = convertToF32Precision
                            ? Float32Type::get(rewriter.getContext())
                            : elemType;
    float accBaseVal = getRedBaseVal(rop);
    auto accBase = rewriter.create<arith::ConstantOp>(
        loc, constantType, rewriter.getFloatAttr(constantType, accBaseVal));
    Value initTensor;

    if (isVectorReduce) {
      // The affine vectorizer cannot vectorize affine loops generated from
      // linalg.reduce for the vector reduce case, so we must rewrite the
      // linalg.reduce to affine loops manually. Here we lower to AllocTensor
      // directly instead of EmptyOp so that the subsequent pass can recognize
      // the patterns (EmptyOp is susceptible to being CSE'd away, making it
      // harder to match the patterns correctly).
      initTensor = rewriter.create<bufferization::AllocTensorOp>(
          loc, RankedTensorType::get({}, constantType), ValueRange{});
      initTensor = rewriter.create<tensor::InsertOp>(loc, accBase, initTensor,
                                                     ValueRange{});
    } else {
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, cast<RankedTensorType>(resType).getShape(), constantType);
      initTensor = rewriter
                       .create<linalg::FillOp>(loc, ValueRange{accBase},
                                               ValueRange{init})
                       .result();
    }

    Value finalResult =
        rewriter
            .create<linalg::ReduceOp>(
                loc, ValueRange{source}, ValueRange{initTensor},
                SmallVector<int64_t>{axis},
                [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                  assert(inputs.size() == 2);
                  Value result =
                      getRedElement(inputs[0], inputs[1], loc, rop, opBuilder,
                                    convertToF32Precision);
                  opBuilder.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);

    if (sourceType.getRank() == 1) {
      finalResult =
          rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
    }

    if (convertToF32Precision) {
      finalResult = rewriter.create<arith::TruncFOp>(
          loc, BFloat16Type::get(rewriter.getContext()), finalResult);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(triton::ReduceOp op,
                  typename triton::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        adaptor.getOperands().front().getType().cast<RankedTensorType>();
    assert(sourceType.hasRank() && "Expected input is "
                                   "ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() &&
           "Expected reduction "
           "axis is within "
           "operand's rank");

    return convertToLinalgReduce(op, adaptor, rewriter);
  }
};

struct GetProgramIDConverter
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;

private:
  const unsigned int LAUNCH_GRID_RANK;

public:
  GetProgramIDConverter(MLIRContext *context, unsigned int launchGridRank)
      : OpConversionPattern(context), LAUNCH_GRID_RANK(launchGridRank) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

// Remove all Meta ops except for AddPtr which is handled by AddPtrConverter.
// Use benefit == 10 to ensure that this pattern always takes precedence over
// other patterns.
struct MetaOpConverter : public RewritePattern {
private:
  // UseAnalysis will tag operations whose results are used only as meta-data
  // with "MetaUse" tag.
  bool isMetaUse(Operation *op) const { return op->hasAttr("MetaUse"); }

public:
  MetaOpConverter(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {

    if (isa<triton::AddPtrOp>(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "AddPtrOp will be handled separately");
    }

    if (isMetaUse(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "requires meta ops");
  }
};

// Convert a pair of cmpf and select to either min or max.
// Leave the pattern as simple as possible because triton has plans to emit
// min and max directly.
struct MinMaxConverter : public OpRewritePattern<arith::CmpFOp> {
  using OpRewritePattern<arith::CmpFOp>::OpRewritePattern;

  MinMaxConverter(MLIRContext *context)
      : OpRewritePattern<arith::CmpFOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(arith::CmpFOp cmpOp,
                                PatternRewriter &rewriter) const final {
    if (!cmpOp.getResult().hasOneUse()) {
      return failure();
    }
    auto selectOp =
        dyn_cast<arith::SelectOp>(*cmpOp.getResult().getUsers().begin());
    if (!selectOp) {
      return failure();
    }

    if (!(cmpOp.getResult() == selectOp.getCondition() &&
          cmpOp.getLhs() == selectOp.getTrueValue() &&
          cmpOp.getRhs() == selectOp.getFalseValue())) {
      return failure();
    }

    auto pred = cmpOp.getPredicate();
    auto loc = cmpOp.getLoc();
    if (pred == arith::CmpFPredicate::OGT) {
      rewriter.replaceOpWithNewOp<arith::MaxFOp>(selectOp, cmpOp.getLhs(),
                                                 cmpOp.getRhs());
    } else if (pred == arith::CmpFPredicate::OLT) {
      rewriter.replaceOpWithNewOp<arith::MinFOp>(selectOp, cmpOp.getLhs(),
                                                 cmpOp.getRhs());
    } else {
      llvm_unreachable("Unhandled predicate");
    }

    rewriter.eraseOp(cmpOp);

    return success();
  }
};

struct DenseConstantConverter : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto attr = cast<DenseElementsAttr>(op.getValue());
    auto loc = op.getLoc();

    auto splatConst = rewriter.create<arith::ConstantOp>(
        loc, attr.getSplatValue<Attribute>(), attr.getElementType());

    auto init = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(op.getResult().getType()).getShape(),
        attr.getElementType());

    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{splatConst},
                                                ValueRange{init});

    return success();
  }
};

} // namespace

void mlir::triton::populateTritonToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MinMaxConverter>(patterns.getContext());
}

void mlir::triton::populateTritonToLinalgConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank) {
  populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
      patterns, typeConverter);
  patterns.add<MetaOpConverter>(patterns.getContext());
  patterns.add<StoreConverter>(patterns.getContext());
  patterns.add<AddPtrConverter>(patterns.getContext());
  patterns.add<GetProgramIDConverter>(patterns.getContext(), launchGridRank);
  patterns.add<YieldConverter>(patterns.getContext());
  patterns.add<LoadConverter>(patterns.getContext());
  patterns.add<LoopConverter>(patterns.getContext());
  patterns.add<BroadcastConverter>(patterns.getContext());
  patterns.add<TransposeConverter>(patterns.getContext());
  patterns.add<MakeRangeConverter>(patterns.getContext());
  patterns.add<ExpandDimsConverter>(patterns.getContext());
  patterns.add<BitcastConverter>(patterns.getContext());
  patterns.add<AssertConverter>(patterns.getContext());
  patterns.add<MatmulConverter>(patterns.getContext());
  patterns.add<SplatConverter>(patterns.getContext());
  patterns.add<ReduceConverter>(patterns.getContext());
  patterns.add<DenseConstantConverter>(patterns.getContext());

  // Note: the ordering here matters!
  // MetaOpConverter has PatternBenefit == 10 which should take precedence over
  // these linalg patterns, but to be safe, add these patterns last so that they
  // will be tried last. Incorrect ordering or having MetaOpConverter has lower
  // PatternBenefit will result in element-wise meta ops being converted to
  // linalg.generic ops.
  linalg::populateElementwiseToLinalgConversionPatterns(patterns);
}
