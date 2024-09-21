#include "OpTypeConversion.h"
#include "TypeConverter.h"

#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTELEMENTWISEOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class ElementwiseOpConversionTarget : public ConversionTarget {
public:
  explicit ElementwiseOpConversionTarget(MLIRContext &ctx,
                                         TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addDynamicallyLegalDialect<arith::ArithDialect>(
        [&](Operation *op) -> std::optional<bool> {
          return converter.isLegal(op);
        });
    addDynamicallyLegalDialect<math::MathDialect>(
        [&](Operation *op) -> std::optional<bool> {
          return converter.isLegal(op);
        });

    addDynamicallyLegalOp<triton::BitcastOp>(
        [](triton::BitcastOp op) { return isa<PointerType>(op.getType()); });
    addIllegalOp<triton::PreciseDivFOp>();
    addIllegalOp<triton::PreciseSqrtOp>();
    addIllegalOp<triton::MulhiUIOp>();
    addIllegalOp<triton::ClampFOp>();
    addIllegalOp<triton::FpToFpOp>();
  }
};

struct ConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(isa<RankedTensorType>(op.getType()));
    auto resTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    assert(resTy);
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValueAttr())) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resTy,
                                                     denseAttr.reshape(resTy));
    } else {
      llvm_unreachable("Unexpected constant attribute");
    }
    return success();
  }
};

struct MulhiUIOpConversion : public OpConversionPattern<triton::MulhiUIOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = rewriter.getRemappedValue(op.getX());
    auto rhs = rewriter.getRemappedValue(op.getY());
    Value res =
        rewriter.create<arith::MulUIExtendedOp>(loc, lhs, rhs).getHigh();
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ClampFOpConversion : public OpConversionPattern<triton::ClampFOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto val = rewriter.getRemappedValue(op.getX());
    auto minVal = rewriter.getRemappedValue(op.getMin());
    auto maxVal = rewriter.getRemappedValue(op.getMax());
    Value res;
    if (op.getPropagateNanAttr().getValue() == PropagateNan::ALL) {
      res = rewriter.create<arith::MaximumFOp>(loc, val, minVal);
      res = rewriter.create<arith::MinimumFOp>(loc, res, maxVal);
    } else {
      res = rewriter.create<arith::MaxNumFOp>(loc, val, minVal);
      res = rewriter.create<arith::MinNumFOp>(loc, res, maxVal);
    }
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct FpToFpOpConversion : public OpConversionPattern<triton::FpToFpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = rewriter.getRemappedValue(op.getSrc());
    auto srcTy = src.getType();
    auto resTy = getTypeConverter()->convertType(op.getType());
    auto srcElemTy = isa<VectorType>(srcTy)
                         ? cast<VectorType>(srcTy).getElementType()
                         : srcTy;
    auto resElemTy = isa<VectorType>(resTy)
                         ? cast<VectorType>(resTy).getElementType()
                         : resTy;

    if (srcElemTy.getIntOrFloatBitWidth() > resElemTy.getIntOrFloatBitWidth()) {
      std::optional<RoundingMode> rounding = op.getRounding();
      assert(rounding && "Rounding mode expected for truncate conversions");
      auto roundingAttr = arith::RoundingModeAttr::get(
          getContext(), *rounding == RoundingMode::RTZ
                            ? arith::RoundingMode::toward_zero
                            : arith::RoundingMode::to_nearest_even);
      rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, resTy, src, roundingAttr,
                                                   nullptr);
      return success();
    }

    if (srcElemTy.getIntOrFloatBitWidth() < resElemTy.getIntOrFloatBitWidth()) {
      rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, resTy, src);
      return success();
    }

    return failure();
  }
};

struct ConvertElementwiseOps
    : public triton::impl::ConvertElementwiseOpsBase<ConvertElementwiseOps> {
  using ConvertElementwiseOpsBase::ConvertElementwiseOpsBase;

  ConvertElementwiseOps() : ConvertElementwiseOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    ElementwiseOpConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);

    patterns.add<ConstantOpConversion>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::ExtSIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::ExtUIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::ExtFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::TruncIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::TruncFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::SIToFPOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::UIToFPOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::FPToSIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::FPToUIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::AddFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::AddIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::SubFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::SubIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MulFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MulIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::DivFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::DivSIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::DivUIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::RemFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::RemSIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::RemUIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::AndIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::OrIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::XOrIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::ShLIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::ShRSIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::ShRUIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::CmpFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::CmpIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::SelectOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MaximumFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MaxNumFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MaxSIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MaxUIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MinimumFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MinNumFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MinSIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<arith::MinUIOp>>(typeConverter, context);

    patterns.add<OpTypeConversion<math::FloorOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::CeilOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::FmaOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::AbsFOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::AbsIOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::ExpOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::Exp2Op>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::ExpM1Op>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::LogOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::Log2Op>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::Log10Op>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::Log1pOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::SinOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::SinhOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::CosOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::CoshOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::TanOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::TanhOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::AcosOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::AcoshOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::AsinOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::AsinhOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::AtanOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::AtanhOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::SqrtOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::RsqrtOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::ErfOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::CbrtOp>>(typeConverter, context);
    patterns.add<OpTypeConversion<math::TruncOp>>(typeConverter, context);

    patterns.add<OpTypeConversion<triton::BitcastOp, arith::BitcastOp>>(
        typeConverter, context);
    patterns.add<OpTypeConversion<triton::PreciseDivFOp, arith::DivFOp>>(
        typeConverter, context);
    patterns.add<OpTypeConversion<triton::PreciseSqrtOp, math::SqrtOp>>(
        typeConverter, context);
    patterns.add<MulhiUIOpConversion>(typeConverter, context);
    patterns.add<ClampFOpConversion>(typeConverter, context);
    patterns.add<FpToFpOpConversion>(typeConverter, context);

    if (failed(applyPartialConversionNoBuildMaterializations(
            mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertElementwiseOps() {
  return std::make_unique<ConvertElementwiseOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
