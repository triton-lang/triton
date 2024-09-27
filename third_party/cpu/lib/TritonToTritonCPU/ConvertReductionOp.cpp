#include "ReduceScanCommon.h"
#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include <numeric>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTREDUCTIONOP
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace cpu
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

struct ReduceOpConversion
    : public ReduceScanOpConversionBase<triton::ReduceOp,
                                        triton::ReduceReturnOp> {
  ReduceOpConversion(bool useReductionOp, bool useMultiDimReductionOp,
                     const TypeConverter &typeConverter, MLIRContext *context)
      : ReduceScanOpConversionBase(typeConverter, context) {

    this->useReductionOp = useReductionOp;
    this->useMultiDimReductionOp = useMultiDimReductionOp;
  }

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // More simple cases with a single input and a single combine operation
    // can be mapped to a vector::MultiDimReductionOp. The resulting code
    // depends on a quality of LLVM backend and is not always perfect though.
    if (succeeded(mapToReductionOp(op, rewriter, useReductionOp,
                                   useMultiDimReductionOp)))
      return success();

    return ReduceScanOpConversionBase::matchAndRewrite(op, adaptor, rewriter);
  }

  SmallVector<Value>
  lower1DInput(ValueRange inputs, ReduceOp op,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();
    int64_t vecSize = cast<VectorType>(inputs[0].getType()).getShape()[0];
    SmallVector<int64_t> range(vecSize);
    std::iota(range.begin(), range.end(), 0);

    SmallVector<Value> dummies = createShuffleDummies(loc, inputs, rewriter);
    SmallVector<Value> res = inputs;
    for (int64_t stride = vecSize / 2; stride > 0; stride = stride / 2) {
      SmallVector<int64_t> shuffleIndices = range;
      for (int64_t i = 0; i < stride; ++i) {
        std::swap(shuffleIndices[i], shuffleIndices[i + stride]);
      }
      SmallVector<Value> shuffledInput;
      for (auto [val, dummy] : llvm::zip(res, dummies)) {
        shuffledInput.push_back(rewriter.create<vector::ShuffleOp>(
            loc, val, dummy, shuffleIndices));
      }

      res = accumulate(shuffledInput, res, combineOp, rewriter);
    }

    // The results are in the first element of each produced vector.
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    for (size_t i = 0; i < res.size(); ++i) {
      res[i] = rewriter.create<vector::ExtractElementOp>(loc, res[i], zero);
    }
    return res;
  }

  SmallVector<Value>
  lowerLeadingDimension(ValueRange inputs, ReduceOp op,
                        ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();
    auto shape = cast<VectorType>(inputs[0].getType()).getShape();
    SmallVector<Value> res;
    for (int64_t idx = 0; idx < shape[0]; ++idx) {
      SmallVector<Value> subInputs(inputs.size());
      std::transform(inputs.begin(), inputs.end(), subInputs.begin(),
                     [&](auto val) {
                       return rewriter.create<vector::ExtractOp>(loc, val, idx);
                     });

      res = accumulate(subInputs, res, combineOp, rewriter);
    }
    return res;
  }

  LogicalResult mapToReductionOp(triton::ReduceOp op,
                                 ConversionPatternRewriter &rewriter,
                                 bool useReductionOp,
                                 bool useMultiDimReductionOp) const {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();

    Value src = rewriter.getRemappedValue(op.getOperand(0));
    VectorType srcTy = cast<VectorType>(src.getType());

    Block *block = op.getBody();
    if (block->getNumArguments() != 2)
      return failure();
    Value accArg = block->getArgument(0);
    Value itArg = block->getArgument(1);

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

    if (useReductionOp && srcTy.getShape().size() == 1) {
      rewriter.replaceOpWithNewOp<vector::ReductionOp>(op, resTy, reductionKind,
                                                       src, acc);
      return success();
    } else if (useMultiDimReductionOp) {
      rewriter.replaceOpWithNewOp<vector::MultiDimReductionOp>(
          op, resTy, reductionKind, src, acc, axis);
      return success();
    }

    return failure();
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
          static_cast<int64_t>(-(1UL << (elemTy.getIntOrFloatBitWidth() - 1))));
    else if (kind == vector::CombiningKind::MINSI)
      initVal = rewriter.getIntegerAttr(
          elemTy, static_cast<int64_t>(
                      (1UL << (elemTy.getIntOrFloatBitWidth() - 1)) - 1));
    else if (kind == vector::CombiningKind::MINIMUMF ||
             kind == vector::CombiningKind::MAXIMUMF) {
      if (elemTy.isF32())
        initVal =
            rewriter.getF32FloatAttr(std::numeric_limits<float>::quiet_NaN());
      else if (elemTy.isF64())
        initVal =
            rewriter.getF64FloatAttr(std::numeric_limits<double>::quiet_NaN());
      else
        llvm_unreachable("Unsupported type for acc init value.");
    }

    else if (kind == vector::CombiningKind::MINNUMF) {
      if (elemTy.isF32())
        initVal =
            rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity());
      else if (elemTy.isF64())
        initVal =
            rewriter.getF64FloatAttr(std::numeric_limits<double>::infinity());
      else
        llvm_unreachable("Unsupported type for acc init value.");
    } else if (kind == vector::CombiningKind::MAXNUMF) {
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

private:
  bool useMultiDimReductionOp;
  bool useReductionOp;
};

struct ConvertReductionOp
    : public triton::cpu::impl::ConvertReductionOpBase<ConvertReductionOp> {
  ConvertReductionOp() = default;

  ConvertReductionOp(bool useReductionOp, bool useMultiDimReductionOp) {
    this->useReductionOp = useReductionOp;
    this->useMultiDimReductionOp = useMultiDimReductionOp;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    ReductionConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<ReduceOpConversion>(useReductionOp, useMultiDimReductionOp,
                                     typeConverter, context);

    if (failed(applyPartialConversionNoBuildMaterializations(
            mod, convTarget, std::move(patterns))))
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

std::unique_ptr<OperationPass<ModuleOp>>
createConvertReductionOp(bool useReductionOp, bool useMultiDimReductionOp) {
  return std::make_unique<ConvertReductionOp>(useReductionOp,
                                              useMultiDimReductionOp);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
