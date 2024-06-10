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
#define GEN_PASS_DEF_CONVERTSCANOP
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class ScanConversionTarget : public ConversionTarget {
public:
  explicit ScanConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addIllegalOp<triton::ScanOp>();
  }
};

struct ScanOpConversion
    : public ReduceScanOpConversionBase<triton::ScanOp, triton::ScanReturnOp> {
  using ReduceScanOpConversionBase::ReduceScanOpConversionBase;

  SmallVector<Value>
  lower1DInput(ValueRange inputs, ScanOp op,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();
    bool reverse = op.getReverse();
    int64_t vecSize = cast<VectorType>(inputs[0].getType()).getShape()[0];
    Type maskTy = VectorType::get(vecSize, rewriter.getI1Type());

    ArrayRef<Value> dummies = createShuffleDummies(loc, inputs, rewriter);
    SmallVector<Value> res = inputs;
    for (int64_t stride = 1; stride < vecSize; stride *= 2) {
      SmallVector<int64_t> shuffleIndices(vecSize, 0);
      int64_t start = reverse ? vecSize - 1 - stride : stride;
      int64_t end = reverse ? -1 : vecSize;
      int64_t step = reverse ? -1 : 1;
      for (int64_t i = start; i != end; i += step) {
        shuffleIndices[i] = i - step * stride;
      }
      SmallVector<Value> shuffledInput;
      for (auto [val, dummy] : llvm::zip(res, dummies)) {
        shuffledInput.push_back(rewriter.create<vector::ShuffleOp>(
            loc, val, dummy, shuffleIndices));
      }

      auto newRes = accumulate(res, shuffledInput, combineOp, rewriter);

      // Number of already computed elements is equal to the current
      // stride. Mask them out using a constant mask.
      SmallVector<bool> maskVals(vecSize, true);
      if (reverse) {
        std::fill(maskVals.rbegin(), maskVals.rbegin() + stride, false);
      } else {
        std::fill(maskVals.begin(), maskVals.begin() + stride, false);
      }
      Value mask = rewriter.create<arith::ConstantOp>(
          loc, maskTy, rewriter.getBoolVectorAttr(maskVals));
      for (size_t i = 0; i < res.size(); ++i) {
        res[i] = vector::selectPassthru(rewriter, mask, newRes[i], res[i]);
      }
    }

    return res;
  }

  SmallVector<Value>
  lowerLeadingDimension(ValueRange inputs, ScanOp op,
                        ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();
    bool reverse = op.getReverse();
    auto shape = cast<VectorType>(inputs[0].getType()).getShape();
    SmallVector<Type> resTypes;
    for (const auto &resTy : op.getResultTypes()) {
      resTypes.push_back(VectorType::get(
          shape, cast<RankedTensorType>(resTy).getElementType()));
    }
    SmallVector<Value> res = makeEmptyResults(loc, resTypes, rewriter);
    SmallVector<Value> acc;
    int64_t start = reverse ? shape[0] - 1 : 0;
    int64_t end = reverse ? -1 : shape[0];
    int64_t step = reverse ? -1 : 1;
    for (int64_t idx = start; idx != end; idx += step) {
      SmallVector<Value> subInputs(inputs.size());
      std::transform(inputs.begin(), inputs.end(), subInputs.begin(),
                     [&](auto val) {
                       return rewriter.create<vector::ExtractOp>(loc, val, idx);
                     });

      acc = accumulate(subInputs, acc, combineOp, rewriter);

      for (size_t i = 0; i < res.size(); ++i) {
        res[i] = rewriter.create<vector::InsertOp>(loc, acc[i], res[i], idx);
      }
    }
    return res;
  }
};

struct ConvertScanOp : public triton::impl::ConvertScanOpBase<ConvertScanOp> {
  using ConvertScanOpBase::ConvertScanOpBase;

  ConvertScanOp() : ConvertScanOpBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    ScanConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<ScanOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertScanOp() {
  return std::make_unique<ConvertScanOp>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
