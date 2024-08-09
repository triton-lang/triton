#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTUNSUPPORTEDOPS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

template <typename OpT>
struct ConvertBf16ToFp32 : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    // TODO: support mixed-type ops?
    if (!isAllBf16(op->getOperandTypes()) || !isAllBf16(op->getResultTypes()))
      return failure();

    Location loc = op.getLoc();
    OperationState newState(loc, OpT::getOperationName());
    // Convert operands to fp32 and generate fp32 op.
    for (auto operand : op->getOperands()) {
      Value newOperand = rewriter.create<arith::ExtFOp>(
          loc, toFp32(operand.getType()), operand);
      newState.operands.push_back(newOperand);
    }
    newState.types = toFp32(op->getResultTypes());
    newState.attributes = op->getAttrs();
    auto newOp = rewriter.create(newState);

    // Convert op results back to Bf16
    SmallVector<Value> results;
    for (auto res : llvm::enumerate(newOp->getResults()))
      results.push_back(rewriter.create<arith::TruncFOp>(
          loc, op->getResult(res.index()).getType(), res.value()));
    rewriter.replaceOp(op, results);

    return success();
  }

  bool isAllBf16(TypeRange types) const {
    return std::all_of(types.begin(), types.end(),
                       [this](auto ty) { return isBf16(ty); });
  }

  SmallVector<Type> toFp32(TypeRange types) const {
    SmallVector<Type> res;
    for (auto ty : types)
      res.push_back(::toFp32(ty));
    return res;
  }
};

template <typename OpT>
struct ConvertIToBf16ToFp32 : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    if (!isBf16(op.getType()))
      return failure();

    Location loc = op.getLoc();
    Value fp32Val =
        rewriter.create<OpT>(loc, toFp32(op.getType()), op.getOperand());
    Value res = rewriter.create<arith::TruncFOp>(loc, op.getType(), fp32Val);
    rewriter.replaceOp(op, res);
    return success();
  }
};

Value convertMemRefToI16(Value memRef, PatternRewriter &rewriter) {
  Value res;
  MemRefType memRefTy = cast<MemRefType>(memRef.getType());
  Type newMemRefTy =
      MemRefType::get(memRefTy.getShape(), rewriter.getI16Type(),
                      memRefTy.getLayout(), memRefTy.getMemorySpace());
  auto insPoint = rewriter.saveInsertionPoint();
  rewriter.setInsertionPointAfter(memRef.getDefiningOp());
  // Memory references for masked operations and transfers are always built
  // with PtrToMemRefOp or ExtractMemRefOp.
  if (auto castOp = memRef.getDefiningOp<PtrToMemRefOp>()) {
    res = rewriter.create<PtrToMemRefOp>(memRef.getLoc(), newMemRefTy,
                                         castOp.getSrc());
  } else {
    auto extractOp = memRef.getDefiningOp<ExtractMemRefOp>();
    assert(extractOp && "Unexpected memref producer");
    res = rewriter.create<ExtractMemRefOp>(memRef.getLoc(), newMemRefTy,
                                           extractOp.getSrc());
  }
  rewriter.restoreInsertionPoint(insPoint);
  return res;
}

struct ConvertBf16MaskedLoadOp : public OpRewritePattern<vector::MaskedLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedLoadOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBf16(op.getType()))
      return failure();

    Location loc = op.getLoc();
    Value newBase = convertMemRefToI16(op.getBase(), rewriter);
    Value newPassThru = rewriter.create<arith::BitcastOp>(
        loc, toInt16(op.getPassThru().getType()), op.getPassThru());
    Value intVal = rewriter.create<vector::MaskedLoadOp>(
        loc, toInt16(op.getType()), newBase, op.getIndices(), op.getMask(),
        newPassThru);
    Value res = rewriter.create<arith::BitcastOp>(loc, op.getType(), intVal);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertBf16MaskedStoreOp
    : public OpRewritePattern<vector::MaskedStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MaskedStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBf16(op.getValueToStore().getType()))
      return failure();

    Location loc = op.getLoc();
    Value newBase = convertMemRefToI16(op.getBase(), rewriter);
    Value intVal = rewriter.create<arith::BitcastOp>(
        loc, toInt16(op.getValueToStore().getType()), op.getValueToStore());
    rewriter.replaceOpWithNewOp<vector::MaskedStoreOp>(
        op, newBase, op.getIndices(), op.getMask(), intVal);
    return success();
  }
};

struct ConvertBf16TransferReadOp
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBf16(op.getType()))
      return failure();

    Location loc = op.getLoc();
    Value newSource = convertMemRefToI16(op.getSource(), rewriter);
    Value newPadding =
        op.getPadding()
            ? rewriter.create<arith::BitcastOp>(
                  loc, toInt16(op.getPadding().getType()), op.getPadding())
            : nullptr;
    Value intVal = rewriter.create<vector::TransferReadOp>(
        loc, toInt16(op.getType()), newSource, op.getIndices(),
        op.getPermutationMapAttr(), newPadding, op.getMask(),
        op.getInBoundsAttr());
    Value res = rewriter.create<arith::BitcastOp>(loc, op.getType(), intVal);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertBf16TransferWriteOp
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBf16(op.getVector().getType()))
      return failure();

    Location loc = op.getLoc();
    Value newSource = convertMemRefToI16(op.getSource(), rewriter);
    Value intVal = rewriter.create<arith::BitcastOp>(
        loc, toInt16(op.getVector().getType()), op.getVector());
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, intVal, newSource, op.getIndices(), op.getPermutationMapAttr(),
        op.getMask(), op.getInBoundsAttr());
    return success();
  }
};

struct ConvertBf16Abs : public OpRewritePattern<math::AbsFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AbsFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isBf16(op.getType()) || !isBf16(op.getOperand().getType()))
      return failure();

    Location loc = op.getLoc();
    Value src = op.getOperand();
    Value intSrc =
        rewriter.create<arith::BitcastOp>(loc, toInt16(op.getType()), src);
    TypedAttr maskAttr = rewriter.getI16IntegerAttr(0x7fff);
    if (auto vecTy = dyn_cast<VectorType>(intSrc.getType()))
      maskAttr = SplatElementsAttr::get(vecTy, maskAttr);
    Value mask = rewriter.create<arith::ConstantOp>(loc, maskAttr);
    Value res = rewriter.create<arith::AndIOp>(loc, intSrc, mask);
    res = rewriter.create<arith::BitcastOp>(loc, op.getType(), res);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertF8Abs : public OpRewritePattern<math::AbsFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::AbsFOp op,
                                PatternRewriter &rewriter) const override {
    if (!isFp8(op.getType()) || !isFp8(op.getOperand().getType()))
      return failure();

    Location loc = op.getLoc();
    Value src = op.getOperand();
    Type srcType = op.getType();

    Value i8Src = op_bitcast(toInt8(srcType), src);
    // Mask out the sign bit
    Value nosign = op_and(i8Src, cst_like(i8Src, 0x7f));
    Value res = op_bitcast(srcType, nosign);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertMixedPrecisionMatmul
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value acc = op.getAcc();
    auto lhsTy = cast<VectorType>(lhs.getType());
    auto rhsTy = cast<VectorType>(rhs.getType());
    auto accTy = cast<VectorType>(acc.getType());
    auto resTy = cast<VectorType>(op.getType());

    if (lhsTy.getElementType() == resTy.getElementType() &&
        rhsTy.getElementType() == resTy.getElementType() &&
        accTy.getElementType() == resTy.getElementType())
      return failure();

    Type commonElemTy = resTy.getElementType();
    if (lhsTy.getElementTypeBitWidth() > commonElemTy.getIntOrFloatBitWidth())
      commonElemTy = lhsTy;
    if (rhsTy.getElementTypeBitWidth() > commonElemTy.getIntOrFloatBitWidth())
      commonElemTy = rhsTy;
    if (accTy.getElementTypeBitWidth() > commonElemTy.getIntOrFloatBitWidth())
      commonElemTy = accTy;

    lhs = castElemTy(loc, lhs, commonElemTy, rewriter);
    rhs = castElemTy(loc, rhs, commonElemTy, rewriter);
    acc = castElemTy(loc, acc, commonElemTy, rewriter);

    Value newRes = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, acc, op.getIndexingMaps(), op.getIteratorTypes());
    newRes = castElemTy(loc, newRes, resTy.getElementType(), rewriter);

    rewriter.replaceOp(op, newRes);
    return success();
  }

  Value castElemTy(Location loc, Value val, Type elemTy,
                   PatternRewriter &rewriter) const {
    auto valTy = cast<VectorType>(val.getType());
    if (valTy.getElementType() == elemTy)
      return val;

    auto resTy = toTyOrVectorOf(valTy, elemTy);
    if (valTy.getElementType().isInteger()) {
      if (valTy.getElementTypeBitWidth() > elemTy.getIntOrFloatBitWidth())
        return rewriter.create<arith::TruncIOp>(loc, resTy, val);
      else
        return rewriter.create<arith::ExtSIOp>(loc, resTy, val);
    } else {
      if (valTy.getElementTypeBitWidth() > elemTy.getIntOrFloatBitWidth())
        return rewriter.create<arith::TruncFOp>(loc, resTy, val);
      else
        return rewriter.create<arith::ExtFOp>(loc, resTy, val);
    }
  }
};

template <typename OpT> struct PromoteOpToFp32 : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  PromoteOpToFp32(MLIRContext *context) : OpRewritePattern<OpT>(context) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    Type opTy = op.getType();

    if (!isFp8(opTy) && !isFp16(opTy) && !isBf16(opTy))
      return failure();

    Type fp32Ty = toFp32(opTy);
    SmallVector<Value> fp32Ops;
    for (auto operand : op->getOperands())
      fp32Ops.push_back(rewriter.create<arith::ExtFOp>(loc, fp32Ty, operand));
    auto newOp = rewriter.create<OpT>(loc, fp32Ty, fp32Ops);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, opTy, newOp);
    return success();
  }
};

struct ConvertUnsupportedOps
    : public triton::cpu::impl::ConvertUnsupportedOpsBase<
          ConvertUnsupportedOps> {
  ConvertUnsupportedOps() = default;

  ConvertUnsupportedOps(bool promoteBf16ToFp32,
                        bool convertMixedPrecisionMatmul,
                        bool promoteLibMathToFp32) {
    this->promoteBf16ToFp32 = promoteBf16ToFp32;
    this->convertMixedPrecisionMatmul = convertMixedPrecisionMatmul;
    this->promoteLibMathToFp32 = promoteLibMathToFp32;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    if (promoteBf16ToFp32) {
      patterns.add<ConvertBf16ToFp32<arith::AddFOp>>(context);
      patterns.add<ConvertBf16ToFp32<arith::SubFOp>>(context);
      patterns.add<ConvertBf16ToFp32<arith::MulFOp>>(context);
      patterns.add<ConvertIToBf16ToFp32<arith::SIToFPOp>>(context);
      patterns.add<ConvertIToBf16ToFp32<arith::UIToFPOp>>(context);
      patterns.add<ConvertBf16MaskedLoadOp>(context);
      patterns.add<ConvertBf16MaskedStoreOp>(context);
      patterns.add<ConvertBf16TransferReadOp>(context);
      patterns.add<ConvertBf16TransferWriteOp>(context);
      patterns.add<ConvertBf16Abs>(context);
    }
    patterns.add<ConvertF8Abs>(context);
    if (convertMixedPrecisionMatmul) {
      patterns.add<ConvertMixedPrecisionMatmul>(context);
    }
    if (promoteLibMathToFp32) {
      patterns.add<PromoteOpToFp32<math::AcosOp>>(context);
      patterns.add<PromoteOpToFp32<math::AcoshOp>>(context);
      patterns.add<PromoteOpToFp32<math::AsinOp>>(context);
      patterns.add<PromoteOpToFp32<math::AsinhOp>>(context);
      patterns.add<PromoteOpToFp32<math::Atan2Op>>(context);
      patterns.add<PromoteOpToFp32<math::AtanOp>>(context);
      patterns.add<PromoteOpToFp32<math::AtanhOp>>(context);
      patterns.add<PromoteOpToFp32<math::CbrtOp>>(context);
      patterns.add<PromoteOpToFp32<math::CeilOp>>(context);
      patterns.add<PromoteOpToFp32<math::CosOp>>(context);
      patterns.add<PromoteOpToFp32<math::CoshOp>>(context);
      patterns.add<PromoteOpToFp32<math::ErfOp>>(context);
      patterns.add<PromoteOpToFp32<math::ExpOp>>(context);
      patterns.add<PromoteOpToFp32<math::Exp2Op>>(context);
      patterns.add<PromoteOpToFp32<math::ExpM1Op>>(context);
      patterns.add<PromoteOpToFp32<math::FloorOp>>(context);
      patterns.add<PromoteOpToFp32<math::FmaOp>>(context);
      patterns.add<PromoteOpToFp32<math::LogOp>>(context);
      patterns.add<PromoteOpToFp32<math::Log2Op>>(context);
      patterns.add<PromoteOpToFp32<math::Log10Op>>(context);
      patterns.add<PromoteOpToFp32<math::Log1pOp>>(context);
      patterns.add<PromoteOpToFp32<math::PowFOp>>(context);
      patterns.add<PromoteOpToFp32<math::RsqrtOp>>(context);
      patterns.add<PromoteOpToFp32<math::SinOp>>(context);
      patterns.add<PromoteOpToFp32<math::SinhOp>>(context);
      patterns.add<PromoteOpToFp32<math::SqrtOp>>(context);
      patterns.add<PromoteOpToFp32<math::TanOp>>(context);
      patterns.add<PromoteOpToFp32<math::TanhOp>>(context);
    }

    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertUnsupportedOps() {
  return std::make_unique<ConvertUnsupportedOps>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertUnsupportedOps(bool promoteBf16ToFp32,
                            bool convertMixedPrecisionMatmul,
                            bool promoteLibMathToFp32) {
  return std::make_unique<ConvertUnsupportedOps>(
      promoteBf16ToFp32, convertMixedPrecisionMatmul, promoteLibMathToFp32);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
