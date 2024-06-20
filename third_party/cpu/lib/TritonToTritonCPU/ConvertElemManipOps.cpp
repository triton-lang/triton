#include "OpTypeConversion.h"
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
#define GEN_PASS_DEF_CONVERTELEMMANIPOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class ElemManipOpConversionTarget : public ConversionTarget {
public:
  explicit ElemManipOpConversionTarget(MLIRContext &ctx,
                                       TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<math::MathDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addIllegalOp<triton::BroadcastOp>();
    addIllegalOp<triton::ExpandDimsOp>();
    addIllegalOp<triton::ReshapeOp>();
    addIllegalOp<triton::TransOp>();
    addIllegalOp<triton::JoinOp>();
    addIllegalOp<triton::CatOp>();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<triton::ReshapeOp> {
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

struct TransOpConversion : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto val = rewriter.getRemappedValue(op.getSrc());
    auto order = op.getOrder();
    SmallVector<int64_t> permutation(order.begin(), order.end());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(op, val, permutation);
    return success();
  }
};

struct JoinOpConversion : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = rewriter.getRemappedValue(op.getLhs());
    auto rhs = rewriter.getRemappedValue(op.getRhs());
    auto interleave = rewriter.create<vector::InterleaveOp>(loc, lhs, rhs);
    // JoinOp creates a new dimension, but InterleaveOp doubles the final one.
    // Use ShapeCastOp to get the required shape.
    auto resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resTy, interleave);
    return success();
  }
};

struct CatOpConversion : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = rewriter.getRemappedValue(op.getLhs());
    auto rhs = rewriter.getRemappedValue(op.getRhs());
    auto lhsTy = dyn_cast<VectorType>(lhs.getType());
    auto rhsTy = dyn_cast<VectorType>(rhs.getType());
    SmallVector<int64_t> indices(lhsTy.getShape()[0] + rhsTy.getShape()[0]);
    std::iota(indices.begin(), indices.end(), 0);
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(op, lhs, rhs, indices);
    return success();
  }
};

struct ConvertElemManipOps
    : public triton::impl::ConvertElemManipOpsBase<ConvertElemManipOps> {
  using ConvertElemManipOpsBase::ConvertElemManipOpsBase;

  ConvertElemManipOps() : ConvertElemManipOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    ElemManipOpConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<OpTypeConversion<triton::BroadcastOp, vector::BroadcastOp>>(
        typeConverter, context);
    patterns.add<OpTypeConversion<triton::ExpandDimsOp, vector::ShapeCastOp>>(
        typeConverter, context);
    patterns.add<ReshapeOpConversion>(typeConverter, context);
    patterns.add<TransOpConversion>(typeConverter, context);
    patterns.add<JoinOpConversion>(typeConverter, context);
    patterns.add<CatOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertElemManipOps() {
  return std::make_unique<ConvertElemManipOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
