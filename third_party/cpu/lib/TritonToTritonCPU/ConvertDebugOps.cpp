#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTDEBUGOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class DebugOpsConversionTarget : public ConversionTarget {
public:
  explicit DebugOpsConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::BuiltinDialect>();
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();

    addLegalOp<arith::ConstantOp>();

    addIllegalOp<triton::PrintOp>();
    addIllegalOp<triton::AssertOp>();
  }
};

struct PrintOpConversion : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // It lowers to triton_cpu.print after converting tensor types to vectors.
    // (tt.print doesn't accept vector types, so we have this intermediate op.)
    if (op.getNumOperands() == 0) {
      rewriter.create<triton::cpu::PrintOp>(loc, op.getPrefix(), op.getHex(),
                                            ValueRange{},
                                            llvm::SmallVector<int, 0>{});
    } else {
      // triton_cpu.print takes up to one vector or scalar operand. It prints
      // each value as a separate print call like the GPU and interpreter.
      assert(op.getNumOperands() == op.getIsSigned().size());
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        Value opr = op.getOperands()[i];
        llvm::SmallVector<int, 1> isSigned = {op.getIsSigned()[i]};
        // TODO: Consider using memrefs for general N-dimensional vectors.
        rewriter.create<triton::cpu::PrintOp>(loc, op.getPrefix(), op.getHex(),
                                              rewriter.getRemappedValue(opr),
                                              isSigned);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct AssertOpConversion : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value acc = rewriter.create<arith::ConstantOp>(loc, i1_ty,
                                                   rewriter.getOneAttr(i1_ty));
    Value condition = rewriter.getRemappedValue(op.getCondition());
    SmallVector<bool> dimsToReduce(
        cast<VectorType>(condition.getType()).getRank(), true);
    condition = rewriter.create<vector::MultiDimReductionOp>(
        loc, condition, acc, dimsToReduce, vector::CombiningKind::AND);
    rewriter.replaceOpWithNewOp<triton::cpu::AssertOp>(op, condition,
                                                       op.getMessage());
    return success();
  }
};

struct ConvertDebugOps
    : public triton::impl::ConvertDebugOpsBase<ConvertDebugOps> {
  using ConvertDebugOpsBase::ConvertDebugOpsBase;

  ConvertDebugOps() : ConvertDebugOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    DebugOpsConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<PrintOpConversion>(typeConverter, context);
    patterns.add<AssertOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDebugOps() {
  return std::make_unique<ConvertDebugOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
