#include "OpTypeConversion.h"
#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#define GEN_PASS_DEF_CONVERTCONTROLFLOWOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class ControlFlowOpConversionTarget : public ConversionTarget {
public:
  explicit ControlFlowOpConversionTarget(MLIRContext &ctx,
                                         TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ForOpConversion : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value lowerBound = rewriter.getRemappedValue(op.getLowerBound());
    Value upperBound = rewriter.getRemappedValue(op.getUpperBound());
    Value step = rewriter.getRemappedValue(op.getStep());
    SmallVector<Value> initArgs;
    if (failed(rewriter.getRemappedValues(op.getInitArgs(), initArgs)))
      return failure();
    // Create new for op with remapped values.
    auto newOp = rewriter.create<scf::ForOp>(op.getLoc(), lowerBound,
                                             upperBound, step, initArgs);
    // Move the old op block and convert its sigature.
    Block *oldBlock = op.getBody();
    Block *newBlock = newOp.getBody();
    rewriter.moveBlockBefore(oldBlock, newOp.getBody());
    rewriter.eraseBlock(newBlock);
    if (failed(rewriter.convertRegionTypes(oldBlock->getParent(),
                                           *getTypeConverter())))
      return failure();
    rewriter.replaceOp(op, newOp);

    return success();
  }
};

struct ConvertControlFlowOps
    : public triton::impl::ConvertControlFlowOpsBase<ConvertControlFlowOps> {
  using ConvertControlFlowOpsBase::ConvertControlFlowOpsBase;

  ConvertControlFlowOps() : ConvertControlFlowOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    ControlFlowOpConversionTarget convTarget(*context, typeConverter);
    convTarget.addDynamicallyLegalOp<scf::YieldOp>(
        [&](Operation *op) -> std::optional<bool> {
          return typeConverter.isLegal(op);
        });
    {
      RewritePatternSet patterns(context);
      patterns.add<OpTypeConversion<scf::YieldOp>>(typeConverter, context);
      if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
        return signalPassFailure();
    }

    convTarget.addDynamicallyLegalOp<scf::ForOp>(
        [&](Operation *op) -> std::optional<bool> {
          return typeConverter.isLegal(op);
        });
    {
      RewritePatternSet patterns(context);
      patterns.add<ForOpConversion>(typeConverter, context);
      if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertControlFlowOps() {
  return std::make_unique<ConvertControlFlowOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
