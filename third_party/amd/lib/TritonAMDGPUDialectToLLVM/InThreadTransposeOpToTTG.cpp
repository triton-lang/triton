#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

struct InThreadTransposeOpConversion
    : public OpConversionPattern<triton::amdgpu::InThreadTransposeOp> {
public:
  explicit InThreadTransposeOpConversion(MLIRContext *ctx,
                                         PatternBenefit benefit)
      : OpConversionPattern<triton::amdgpu::InThreadTransposeOp>(ctx, benefit) {
  }

  LogicalResult
  matchAndRewrite(triton::amdgpu::InThreadTransposeOp op, OpAdaptor adaptor,
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

void populateInThreadTransposeOpToTTGPatterns(MLIRContext *ctx,
                                              RewritePatternSet &patterns,
                                              PatternBenefit benefit) {
  patterns.add<InThreadTransposeOpConversion>(ctx, benefit);
}

} // namespace mlir::triton::AMD
