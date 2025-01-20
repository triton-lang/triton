#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace ::mlir::triton::gpu;

namespace {
class GenericFMAVectorMultiplier : public FMAVectorMultiplier {
  ConversionPatternRewriter &rewriter;
  Location loc;

public:
  GenericFMAVectorMultiplier(ConversionPatternRewriter &rewriter, Location loc)
      : rewriter(rewriter), loc(loc) {}

  mlir::Value multiplyVectors(mlir::ArrayRef<mlir::Value> a,
                              mlir::ArrayRef<mlir::Value> b,
                              mlir::Value c) override {
    auto K = a.size();
    assert(b.size() == K);
    mlir::Value accum = c;
    for (int k = 0; k < K; ++k)
      accum = rewriter.create<LLVM::FMulAddOp>(loc, a[k], b[k], accum);
    return accum;
  }
};

} // namespace

LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter) {
  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();
  GenericFMAVectorMultiplier multiplier(rewriter, loc);
  return parametricConvertFMADot(op, adaptor, typeConverter, rewriter,
                                 multiplier);
}
