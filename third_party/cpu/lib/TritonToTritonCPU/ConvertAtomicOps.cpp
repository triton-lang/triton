#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTATOMICOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class AtomicConversionTarget : public ConversionTarget {
public:
  explicit AtomicConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addDynamicallyLegalOp<triton::AtomicRMWOp>(
        [&](triton::AtomicRMWOp op) -> std::optional<bool> {
          return converter.isLegal(op) && !op.getMask();
        });
    addDynamicallyLegalOp<triton::AtomicCASOp>(
        [&](triton::AtomicCASOp op) -> std::optional<bool> {
          return converter.isLegal(op);
        });
  }
};

struct AtomicRMWOpConversion : public OpConversionPattern<triton::AtomicRMWOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto mask =
        op.getMask() ? rewriter.getRemappedValue(op.getMask()) : nullptr;
    arith::ConstantOp maskCst = mask ? getConstMaskDef(mask) : nullptr;
    auto rmwOp = op.getAtomicRmwOp();
    auto ptrs = rewriter.getRemappedValue(op.getPtr());
    auto vals = rewriter.getRemappedValue(op.getVal());
    auto sem = op.getSem();
    auto scope = op.getScope();

    if (mask && !isa<VectorType>(mask.getType())) {
      auto res = lowerScalarMaskToCF(loc, rmwOp, ptrs, vals, mask, sem, scope,
                                     rewriter);
      rewriter.replaceOp(op, res);
      return success();
    }

    auto ptrTy = cast<RankedTensorType>(op.getPtr().getType()).getElementType();
    auto vecTy = cast<VectorType>(vals.getType());
    auto strides = computeStrides(vecTy.getShape());
    auto res =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(vecTy));
    int64_t numElems = vecTy.getNumElements();
    for (int64_t idx = 0; idx < numElems; ++idx) {
      auto indices = delinearize(idx, strides);
      Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, indices);
      ptr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
      Value val = rewriter.create<vector::ExtractOp>(loc, vals, indices);
      Value resElem;

      if (mask && !maskCst) {
        // Non-const mask values are lowered to CF.
        Value maskVal = rewriter.create<vector::ExtractOp>(loc, mask, indices);
        resElem = lowerScalarMaskToCF(loc, rmwOp, ptr, val, maskVal, sem, scope,
                                      rewriter);
      } else if (!mask ||
                 (maskCst && cast<DenseElementsAttr>(maskCst.getValue())
                                 .getValues<bool>()[idx])) {
        // Const true mask case.
        resElem = rewriter.create<triton::AtomicRMWOp>(
            loc, val.getType(), rmwOp, ptr, val, nullptr, sem, scope);
      }

      // Elements with const false mask are skipped.
      if (resElem) {
        rewriter.create<vector::InsertOp>(loc, resElem, res, indices);
      }
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  Value lowerScalarMaskToCF(Location loc, RMWOp rmwOp, Value ptr, Value val,
                            Value mask, MemSemantic sem, MemSyncScope scope,
                            ConversionPatternRewriter &rewriter) const {
    // Check for constant mask.
    if (auto maskDef = mask.getDefiningOp<arith::ConstantOp>()) {
      auto maskVal = cast<IntegerAttr>(maskDef.getValue());
      if (maskVal.getValue().isZero()) {
        return rewriter.create<arith::ConstantOp>(
            loc, rewriter.getZeroAttr(val.getType()));
      } else {
        return rewriter.create<triton::AtomicRMWOp>(
            loc, val.getType(), rmwOp, ptr, val, nullptr, sem, scope);
      }
    }

    Block *headerBlock = rewriter.getBlock();
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(val.getType()));
    Block *condBlock =
        rewriter.splitBlock(headerBlock, rewriter.getInsertionPoint());
    rewriter.setInsertionPointToStart(condBlock);
    Value resVal = rewriter.create<triton::AtomicRMWOp>(
        loc, val.getType(), rmwOp, ptr, val, nullptr, sem, scope);
    Block *footerBlock =
        rewriter.splitBlock(condBlock, rewriter.getInsertionPoint());
    Value res = footerBlock->addArgument(resVal.getType(), resVal.getLoc());
    rewriter.setInsertionPointToEnd(headerBlock);
    rewriter.create<cf::CondBranchOp>(loc, mask, condBlock, footerBlock, zero);
    rewriter.setInsertionPointToEnd(condBlock);
    rewriter.create<cf::BranchOp>(loc, footerBlock, resVal);
    rewriter.setInsertionPointToStart(footerBlock);

    return res;
  }

  arith::ConstantOp getConstMaskDef(Value mask) const {
    while (auto cast = mask.getDefiningOp<UnrealizedConversionCastOp>())
      mask = cast.getOperand(0);
    return mask.getDefiningOp<arith::ConstantOp>();
  }
};

struct AtomicCASOpConversion : public OpConversionPattern<triton::AtomicCASOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptrs = rewriter.getRemappedValue(op.getPtr());
    auto cmpVals = rewriter.getRemappedValue(op.getCmp());
    auto vals = rewriter.getRemappedValue(op.getVal());
    auto sem = op.getSem();
    auto scope = op.getScope();
    auto ptrTy = cast<RankedTensorType>(op.getPtr().getType()).getElementType();
    auto vecTy = cast<VectorType>(vals.getType());
    auto strides = computeStrides(vecTy.getShape());
    auto res =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(vecTy));
    int64_t numElems = vecTy.getNumElements();
    for (int64_t idx = 0; idx < numElems; ++idx) {
      auto indices = delinearize(idx, strides);
      Value ptr = rewriter.create<vector::ExtractOp>(loc, ptrs, indices);
      ptr = rewriter.create<IntToPtrOp>(loc, ptrTy, ptr);
      Value val = rewriter.create<vector::ExtractOp>(loc, vals, indices);
      Value cmpVal = rewriter.create<vector::ExtractOp>(loc, cmpVals, indices);
      Value resElem = rewriter.create<triton::AtomicCASOp>(
          loc, val.getType(), ptr, cmpVal, val, sem, scope);
      rewriter.create<vector::InsertOp>(loc, resElem, res, indices);
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct ConvertAtomicOps
    : public triton::impl::ConvertAtomicOpsBase<ConvertAtomicOps> {
  using ConvertAtomicOpsBase::ConvertAtomicOpsBase;

  ConvertAtomicOps() : ConvertAtomicOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    AtomicConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<AtomicRMWOpConversion>(typeConverter, context);
    patterns.add<AtomicCASOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertAtomicOps() {
  return std::make_unique<ConvertAtomicOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
