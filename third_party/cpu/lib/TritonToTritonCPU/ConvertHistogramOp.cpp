#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTHISTOGRAMOP
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class HistogramConversionTarget : public ConversionTarget {
public:
  explicit HistogramConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::BuiltinDialect>();
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<math::MathDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();

    addIllegalOp<triton::HistogramOp>();
  }
};

struct HistogramOpConversion : public OpConversionPattern<triton::HistogramOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = rewriter.getRemappedValue(op.getSrc());
    auto srcTy = dyn_cast<VectorType>(src.getType());
    auto resTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(op.getType()));

    if (srcTy.getRank() != 1)
      llvm_unreachable("unsupported input for histogram op (rank != 1)");

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, resTy, rewriter.getZeroAttr(resTy));
    Value one = rewriter.create<arith::ConstantOp>(loc, resTy,
                                                   rewriter.getOneAttr(resTy));
    VectorType cmpVecTy =
        VectorType::get(resTy.getShape(), srcTy.getElementType());
    Value rangeVec = rewriter.create<arith::ConstantOp>(
        loc, resTy, makeRangeAttr(cmpVecTy, rewriter));
    Value res = zero;
    for (int64_t i = 0; i < srcTy.getShape()[0]; ++i) {
      Value idx = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getIndexType(), rewriter.getIndexAttr(i));
      Value elem = rewriter.create<vector::ExtractElementOp>(loc, src, idx);
      Value elemVec = rewriter.create<vector::BroadcastOp>(loc, cmpVecTy, elem);
      Value mask = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                  elemVec, rangeVec);
      Value delta = vector::selectPassthru(rewriter, mask, one, zero);
      res = rewriter.create<arith::AddIOp>(loc, res, delta);
    }

    rewriter.replaceOp(op, res);

    return success();
  }

  TypedAttr makeRangeAttr(VectorType resTy,
                          ConversionPatternRewriter &rewriter) const {
    Type elemTy = resTy.getElementType();
    if (elemTy.isInteger(32)) {
      SmallVector<int32_t> range(resTy.getShape()[0]);
      std::iota(range.begin(), range.end(), 0);
      return rewriter.getI32VectorAttr(range);
    } else if (elemTy.isInteger(64)) {
      SmallVector<int64_t> range(resTy.getShape()[0]);
      std::iota(range.begin(), range.end(), 0);
      return rewriter.getI64VectorAttr(range);
    } else {
      llvm_unreachable(
          "unsupported src elem type for histogram (expected i32 or i64)");
    }
  }
};

struct ConvertHistogramOp
    : public triton::impl::ConvertHistogramOpBase<ConvertHistogramOp> {
  using ConvertHistogramOpBase::ConvertHistogramOpBase;

  ConvertHistogramOp() : ConvertHistogramOpBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    HistogramConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<HistogramOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertHistogramOp() {
  return std::make_unique<ConvertHistogramOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
