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

// This is borrowed from SCFWhilePattern in
//    lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp
class WhileOpConversion : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), newResultTypes,
                                               adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
//    and
//    lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
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
    convTarget.addDynamicallyLegalOp<scf::ConditionOp>(
        [&](Operation *op) -> std::optional<bool> {
          return typeConverter.isLegal(op);
        });
    {
      RewritePatternSet patterns(context);
      patterns.add<OpTypeConversion<scf::YieldOp>>(typeConverter, context);
      patterns.add<OpTypeConversion<scf::ConditionOp>>(typeConverter, context);
      if (failed(applyPartialConversionNoBuildMaterializations(
              mod, convTarget, std::move(patterns))))
        return signalPassFailure();
    }

    convTarget.addDynamicallyLegalOp<scf::IfOp>(
        [&](Operation *op) -> std::optional<bool> {
          return typeConverter.isLegal(op);
        });
    {
      RewritePatternSet patterns(context);
      patterns.add<SCFIfPattern>(typeConverter, context);
      if (failed(applyPartialConversionNoBuildMaterializations(
              mod, convTarget, std::move(patterns))))
        return signalPassFailure();
    }

    convTarget.addDynamicallyLegalOp<scf::ForOp>(
        [&](Operation *op) -> std::optional<bool> {
          return typeConverter.isLegal(op);
        });
    convTarget.addDynamicallyLegalOp<scf::WhileOp>(
        [&](Operation *op) -> std::optional<bool> {
          return typeConverter.isLegal(op);
        });
    {
      RewritePatternSet patterns(context);
      patterns.add<ForOpConversion>(typeConverter, context);
      patterns.add<WhileOpConversion>(typeConverter, context);
      if (failed(applyPartialConversionNoBuildMaterializations(
              mod, convTarget, std::move(patterns))))
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
