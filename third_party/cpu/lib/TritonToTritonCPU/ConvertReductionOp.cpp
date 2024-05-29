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
#define GEN_PASS_DEF_CONVERTREDUCTIONOP
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class ReductionConversionTarget : public ConversionTarget {
public:
  explicit ReductionConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addIllegalOp<triton::ReduceOp>();
  }
};

struct ReduceOpConversion : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = op.getContext();
    // Currently, only simple reductions with a single input argumet are
    // supported.
    // TODO: support generic case.
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();

    Value src = rewriter.getRemappedValue(op.getOperand(0));
    VectorType srcTy = dyn_cast<VectorType>(src.getType());
    assert(srcTy);

    Block *block = op.getBody();
    if (block->getNumArguments() != 2)
      return failure();
    Value itArg = block->getArgument(0);
    Value accArg = block->getArgument(1);

    auto &blockOps = block->getOperations();
    if (blockOps.size() != 2)
      return failure();

    Operation &retOp = blockOps.back();
    if (!isa<ReduceReturnOp>(retOp) || retOp.getNumOperands() != 1)
      return failure();

    Value retVal = retOp.getOperand(0);
    Operation *defOp = retVal.getDefiningOp();
    if (!defOp || defOp->getNumOperands() != 2)
      return failure();

    Value lhs = defOp->getOperand(0);
    Value rhs = defOp->getOperand(1);
    if ((lhs != itArg || rhs != accArg) && (lhs != accArg || rhs != itArg))
      return failure();

    vector::CombiningKind reductionKind;
    if (failed(detectReductionKind(defOp, reductionKind)))
      return failure();

    Type resTy = getTypeConverter()->convertType(op.getType(0));
    Value acc = buildInitValue(op.getLoc(), resTy, reductionKind, rewriter);
    int64_t axis = op.getAxis();
    rewriter.replaceOpWithNewOp<vector::MultiDimReductionOp>(
        op, resTy, reductionKind, src, acc, rewriter.getI64ArrayAttr(axis));
    return success();
  }

  LogicalResult detectReductionKind(Operation *op,
                                    vector::CombiningKind &out) const {
    if (isa<arith::AddFOp, arith::AddIOp>(op))
      out = vector::CombiningKind::ADD;
    else if (isa<arith::MulFOp, arith::MulIOp>(op))
      out = vector::CombiningKind::MUL;
    else if (isa<arith::MinSIOp>(op))
      out = vector::CombiningKind::MINSI;
    else if (isa<arith::MinUIOp>(op))
      out = vector::CombiningKind::MINUI;
    else if (isa<arith::MinimumFOp>(op))
      out = vector::CombiningKind::MINIMUMF;
    else if (isa<arith::MinNumFOp>(op))
      out = vector::CombiningKind::MINNUMF;
    else if (isa<arith::MaxSIOp>(op))
      out = vector::CombiningKind::MAXSI;
    else if (isa<arith::MaxUIOp>(op))
      out = vector::CombiningKind::MAXUI;
    else if (isa<arith::MaximumFOp>(op))
      out = vector::CombiningKind::MAXIMUMF;
    else if (isa<arith::MaxNumFOp>(op))
      out = vector::CombiningKind::MAXNUMF;
    else if (isa<arith::AndIOp>(op))
      out = vector::CombiningKind::AND;
    else if (isa<arith::OrIOp>(op))
      out = vector::CombiningKind::OR;
    else if (isa<arith::XOrIOp>(op))
      out = vector::CombiningKind::XOR;
    else
      return failure();
    return success();
  }

  Value buildInitValue(Location loc, Type resTy, vector::CombiningKind kind,
                       ConversionPatternRewriter &rewriter) const {
    VectorType vecTy = dyn_cast<VectorType>(resTy);
    Type elemTy = vecTy ? vecTy.getElementType() : resTy;

    TypedAttr initVal;
    if (kind == vector::CombiningKind::ADD ||
        kind == vector::CombiningKind::OR ||
        kind == vector::CombiningKind::XOR ||
        kind == vector::CombiningKind::MAXUI)
      initVal = rewriter.getZeroAttr(elemTy);
    else if (kind == vector::CombiningKind::MUL)
      initVal = rewriter.getOneAttr(elemTy);
    else if (kind == vector::CombiningKind::AND ||
             kind == vector::CombiningKind::MINUI)
      initVal = rewriter.getIntegerAttr(elemTy, -1);
    else if (kind == vector::CombiningKind::MAXSI)
      initVal = rewriter.getIntegerAttr(
          elemTy,
          static_cast<int64_t>(1UL << (elemTy.getIntOrFloatBitWidth() - 1)));
    else if (kind == vector::CombiningKind::MINSI)
      initVal = rewriter.getIntegerAttr(
          elemTy, static_cast<int64_t>(
                      1UL << (elemTy.getIntOrFloatBitWidth() - 1) - 1));
    else if (kind == vector::CombiningKind::MINIMUMF ||
             kind == vector::CombiningKind::MINNUMF) {
      if (elemTy.isF32())
        initVal =
            rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity());
      else if (elemTy.isF64())
        initVal =
            rewriter.getF64FloatAttr(std::numeric_limits<double>::infinity());
      else
        llvm_unreachable("Unsupported type for acc init value.");
    } else if (kind == vector::CombiningKind::MAXIMUMF ||
               kind == vector::CombiningKind::MAXNUMF) {
      if (elemTy.isF32())
        initVal =
            rewriter.getF32FloatAttr(-std::numeric_limits<float>::infinity());
      else if (elemTy.isF64())
        initVal =
            rewriter.getF64FloatAttr(-std::numeric_limits<double>::infinity());
      else
        llvm_unreachable("Unsupported type for acc init value.");
    }

    if (vecTy)
      initVal = SplatElementsAttr::get(vecTy, initVal);

    return rewriter.create<arith::ConstantOp>(loc, resTy, initVal);
  }
};

struct ConvertReductionOp
    : public triton::impl::ConvertReductionOpBase<ConvertReductionOp> {
  using ConvertReductionOpBase::ConvertReductionOpBase;

  ConvertReductionOp() : ConvertReductionOpBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    ReductionConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<ReduceOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertReductionOp() {
  return std::make_unique<ConvertReductionOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
