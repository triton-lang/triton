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

    addIllegalOp<triton::PrintOp>();
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
                                            ValueRange{});
    } else {
      // triton_cpu.print takes up to one vector or scalar operand. It prints
      // each value as a separate print call like the GPU and interpreter.
      for (size_t i = 0; i < op.getNumOperands(); i++) {
        Value opr = op.getOperands()[i];
        // TODO: Consider using memrefs for general N-dimensional vectors.
        rewriter.create<triton::cpu::PrintOp>(loc, op.getPrefix(), op.getHex(),
                                              rewriter.getRemappedValue(opr));
      }
    }

    rewriter.eraseOp(op);
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
