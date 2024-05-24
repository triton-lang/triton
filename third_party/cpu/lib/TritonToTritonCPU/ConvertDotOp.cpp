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
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTDOTOP
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class DotConversionTarget : public ConversionTarget {
public:
  explicit DotConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addIllegalOp<triton::DotOp>();
  }
};

struct DotOpConversion : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    Value a = rewriter.getRemappedValue(op.getA());
    Value b = rewriter.getRemappedValue(op.getB());
    Value c = rewriter.getRemappedValue(op.getC());
    auto aMap = AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx);
    auto bMap = AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx);
    auto cMap = AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx);
    auto iteratorTypes = rewriter.getArrayAttr(
        {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, a, b, c, rewriter.getAffineMapArrayAttr({aMap, bMap, cMap}),
        iteratorTypes);
    return success();
  }
};

struct ConvertDotOp : public triton::impl::ConvertDotOpBase<ConvertDotOp> {
  using ConvertDotOpBase::ConvertDotOpBase;

  ConvertDotOp() : ConvertDotOpBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    DotConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<DotOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotOp() {
  return std::make_unique<ConvertDotOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
