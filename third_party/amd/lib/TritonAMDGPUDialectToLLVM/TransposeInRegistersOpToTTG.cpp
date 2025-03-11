#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

struct TransposeInRegistersOpConversion
    : public OpConversionPattern<triton::amdgpu::TransposeInRegistersOp> {
public:
  explicit TransposeInRegistersOpConversion(MLIRContext *ctx,
                                            PatternBenefit benefit)
      : OpConversionPattern<triton::amdgpu::TransposeInRegistersOp>(ctx,
                                                                    benefit) {}

  LogicalResult
  matchAndRewrite(triton::amdgpu::TransposeInRegistersOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ctx = getContext();
    auto dstType = op.getResult().getType();
    auto src = op.getSrc();
    rewriter.replaceOpWithNewOp<ttg::ConvertLayoutOp>(op, dstType, src);
    return success();
  }
};

} // namespace

namespace mlir::triton::AMD {

void populateTransposeInRegistersOpToTTGPatterns(MLIRContext *ctx,
                                                 RewritePatternSet &patterns,
                                                 PatternBenefit benefit) {
  patterns.add<TransposeInRegistersOpConversion>(ctx, benefit);
}

} // namespace mlir::triton::AMD
