#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
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

    addIllegalOp<triton::BitcastOp>();
    addIllegalOp<triton::BroadcastOp>();
    addIllegalOp<triton::ExpandDimsOp>();
    addIllegalOp<triton::PreciseDivFOp>();
    addIllegalOp<triton::PreciseSqrtOp>();
    addIllegalOp<triton::ReshapeOp>();
  }
};

template <typename OpT, typename ResOpT = OpT>
struct ElementwiseOpConversion : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpConversionPattern<OpT>::getTypeConverter;
  using typename OpConversionPattern<OpT>::OpAdaptor;

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    OperationState newState(op.getLoc(), ResOpT::getOperationName());
    // Convert operands.
    for (auto operand : op->getOperands()) {
      Value newOperand = rewriter.getRemappedValue(operand);
      newState.operands.push_back(newOperand);
    }
    // Convert result types.
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                newState.types))) {
      return failure();
    }
    newState.attributes = op->getAttrs();

    auto newOp = rewriter.create(newState);
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

template <>
struct ElementwiseOpConversion<arith::ConstantOp>
    : public OpConversionPattern<arith::ConstantOp> {
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

template <>
struct ElementwiseOpConversion<triton::ReshapeOp>
    : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(isa<RankedTensorType>(op.getType()));
    auto loc = op.getLoc();
    auto src = rewriter.getRemappedValue(op.getSrc());
    auto srcShape = dyn_cast<VectorType>(src.getType()).getShape();
    auto resTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    auto dstShape = resTy.getShape();
    auto elemTy = resTy.getElementType();

    // There are restrictions on how shape can be modified by ShapeCastOp
    // when rank is changed. For now, we simply detect it and handle through
    // a cast to 1D vector. Better solution may be required later.
    if (canCastShape(srcShape, dstShape)) {
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
          op, VectorType::get(dstShape, elemTy), src);
    } else {
      SmallVector<int64_t> tmpShape({resTy.getNumElements()});
      auto tmp = rewriter.create<vector::ShapeCastOp>(
          loc, VectorType::get(tmpShape, elemTy), src);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
          op, VectorType::get(dstShape, elemTy), tmp);
    }
    return success();
  }

private:
  bool canCastShape(ArrayRef<int64_t> src, ArrayRef<int64_t> dst) const {
    if (src.size() == dst.size())
      return true;
    if (src.size() > dst.size())
      return canCastShape(dst, src);

    size_t srcIdx = 0;
    size_t dstIdx = 0;
    while (srcIdx < src.size() && dstIdx < dst.size()) {
      if (src[srcIdx] == 1) {
        ++srcIdx;
      } else {
        // Source dim size should be a product of continuous dest dim sizes.
        int64_t srcSize = src[srcIdx++];
        int64_t dstSize = dst[dstIdx++];
        while (dstSize < srcSize && dstIdx < dst.size())
          dstSize *= dst[dstIdx++];
        if (dstSize != srcSize)
          return false;
      }
    }

    // Skip trailing 1s.
    while (srcIdx < src.size() && src[srcIdx] == 1)
      ++srcIdx;
    while (dstIdx < dst.size() && dst[dstIdx] == 1)
      ++dstIdx;

    return srcIdx == src.size() && dstIdx == dst.size();
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

    patterns.add<ElementwiseOpConversion<arith::ConstantOp>>(typeConverter,
                                                             context);
    patterns.add<ElementwiseOpConversion<arith::ExtSIOp>>(typeConverter,
                                                          context);
    patterns.add<ElementwiseOpConversion<arith::ExtUIOp>>(typeConverter,
                                                          context);
    patterns.add<ElementwiseOpConversion<arith::ExtFOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::TruncIOp>>(typeConverter,
                                                           context);
    patterns.add<ElementwiseOpConversion<arith::TruncFOp>>(typeConverter,
                                                           context);
    patterns.add<ElementwiseOpConversion<arith::SIToFPOp>>(typeConverter,
                                                           context);
    patterns.add<ElementwiseOpConversion<arith::UIToFPOp>>(typeConverter,
                                                           context);
    patterns.add<ElementwiseOpConversion<arith::FPToSIOp>>(typeConverter,
                                                           context);
    patterns.add<ElementwiseOpConversion<arith::FPToUIOp>>(typeConverter,
                                                           context);
    patterns.add<ElementwiseOpConversion<arith::AddFOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::AddIOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::SubFOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::SubIOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::MulFOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::MulIOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::DivFOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::DivSIOp>>(typeConverter,
                                                          context);
    patterns.add<ElementwiseOpConversion<arith::DivUIOp>>(typeConverter,
                                                          context);
    patterns.add<ElementwiseOpConversion<arith::RemFOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::RemSIOp>>(typeConverter,
                                                          context);
    patterns.add<ElementwiseOpConversion<arith::RemUIOp>>(typeConverter,
                                                          context);
    patterns.add<ElementwiseOpConversion<arith::AndIOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::OrIOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<arith::XOrIOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::ShLIOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::ShRSIOp>>(typeConverter,
                                                          context);
    patterns.add<ElementwiseOpConversion<arith::ShRUIOp>>(typeConverter,
                                                          context);
    patterns.add<ElementwiseOpConversion<arith::CmpFOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::CmpIOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<arith::SelectOp>>(typeConverter,
                                                           context);

    patterns.add<ElementwiseOpConversion<math::FloorOp>>(typeConverter,
                                                         context);
    patterns.add<ElementwiseOpConversion<math::CeilOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::FmaOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::AbsFOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::AbsIOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::ExpOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::Exp2Op>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::LogOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::Log2Op>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::SinOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::CosOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::SqrtOp>>(typeConverter, context);
    patterns.add<ElementwiseOpConversion<math::ErfOp>>(typeConverter, context);

    patterns.add<ElementwiseOpConversion<triton::BitcastOp, arith::BitcastOp>>(
        typeConverter, context);
    patterns
        .add<ElementwiseOpConversion<triton::BroadcastOp, vector::BroadcastOp>>(
            typeConverter, context);
    patterns.add<
        ElementwiseOpConversion<triton::ExpandDimsOp, vector::ShapeCastOp>>(
        typeConverter, context);
    patterns.add<ElementwiseOpConversion<triton::PreciseDivFOp, arith::DivFOp>>(
        typeConverter, context);
    patterns.add<ElementwiseOpConversion<triton::PreciseSqrtOp, math::SqrtOp>>(
        typeConverter, context);
    patterns.add<ElementwiseOpConversion<triton::ReshapeOp>>(typeConverter,
                                                             context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
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
