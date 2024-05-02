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
#define GEN_PASS_DEF_CONVERTPTROPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

unsigned getElemBitWidth(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.getElementType().getIntOrFloatBitWidth();
  if (auto vectorTy = dyn_cast<VectorType>(type))
    return vectorTy.getElementType().getIntOrFloatBitWidth();
  return type.getIntOrFloatBitWidth();
}

class PtrConversionTarget : public ConversionTarget {
public:
  explicit PtrConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Allow only scalar pointer conversion.
    addDynamicallyLegalOp<triton::PtrToIntOp>(
        [](triton::PtrToIntOp op) { return op.getType().isInteger(); });
    addDynamicallyLegalOp<triton::IntToPtrOp>([](triton::IntToPtrOp op) {
      return op.getSrc().getType().isInteger();
    });
  }
};

struct MakeRangeOpConversion : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t start = static_cast<int32_t>(op.getStart());
    int32_t end = static_cast<int32_t>(op.getEnd());
    assert(end >= start);

    llvm::SmallVector<int32_t> values;
    values.reserve(end - start);
    for (int32_t v = start; v < end; ++v) {
      values.push_back(v);
    }

    Type resTy = getTypeConverter()->convertType(op.getType());
    auto newOp = rewriter.create<arith::ConstantOp>(
        op.getLoc(), resTy, rewriter.getI32VectorAttr(values));

    rewriter.replaceOp(op, newOp);
    return success();
  }
};

struct SplatOpConversion : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value val = op.getSrc();
    Type dstValType = getTypeConverter()->convertType(val.getType());
    // Cast pointer
    if (isa<PointerType>(val.getType()))
      val = rewriter
                .create<PtrToIntOp>(
                    loc, getTypeConverter()->convertType(val.getType()), val)
                .getResult();
    Type resType = getTypeConverter()->convertType(op.getType());
    auto cast = rewriter.create<vector::SplatOp>(loc, resType, val);

    rewriter.replaceOp(op, cast);
    return success();
  }
};

struct AddPtrOpConversion : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value ptr = rewriter.getRemappedValue(op.getPtr());
    Value offset = rewriter.getRemappedValue(op.getOffset());
    unsigned offsetBitWidth = getElemBitWidth(offset.getType());
    unsigned elemBitWidth = getPointeeBitWidth(op.getPtr().getType());
    // Compute scale. i1 elements take 1 byte.
    Value scale = rewriter.create<arith::ConstantIntOp>(
        loc, (elemBitWidth + 7) / 8, offsetBitWidth);
    if (isa<VectorType>(offset.getType()))
      scale = rewriter.create<vector::SplatOp>(loc, offset.getType(), scale);
    offset = rewriter.create<arith::MulIOp>(loc, offset, scale);
    offset = rewriter.create<arith::ExtSIOp>(loc, ptr.getType(), offset);
    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, ptr.getType(), ptr, offset);
    return success();
  }
};

struct PtrToIntOpConversion : public OpConversionPattern<triton::PtrToIntOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value val = rewriter.getRemappedValue(op.getSrc());
    auto resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<vector::BitCastOp>(op, resTy, val);
    return success();
  }
};

struct IntToPtrOpConversion : public OpConversionPattern<triton::IntToPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::IntToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value val = rewriter.getRemappedValue(op.getSrc());
    auto resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<vector::BitCastOp>(op, resTy, val);
    return success();
  }
};

struct ConvertPtrOps : public triton::impl::ConvertPtrOpsBase<ConvertPtrOps> {
  using ConvertPtrOpsBase::ConvertPtrOpsBase;

  ConvertPtrOps() : ConvertPtrOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    PtrConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<MakeRangeOpConversion>(typeConverter, context);
    patterns.add<SplatOpConversion>(typeConverter, context);
    patterns.add<AddPtrOpConversion>(typeConverter, context);
    patterns.add<PtrToIntOpConversion>(typeConverter, context);
    patterns.add<IntToPtrOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertPtrOps() {
  return std::make_unique<ConvertPtrOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
