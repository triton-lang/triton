#include "OptCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DECOMPOSEFPCONVERSIONS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

struct Fp32ToBf16Conversion : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp op,
                                PatternRewriter &rewriter) const override {
    Value src = op.getIn();
    if (!isBf16(op.getType()) || !isFp32(src.getType()))
      return failure();

    Location loc = op.getLoc();
    Value i32Src =
        rewriter.create<arith::BitcastOp>(loc, toInt32(src.getType()), src);
    TypedAttr shiftValAttr = rewriter.getI32IntegerAttr(16);
    if (auto vecTy = dyn_cast<VectorType>(i32Src.getType()))
      shiftValAttr = SplatElementsAttr::get(vecTy, shiftValAttr);
    Value shiftedSrc = rewriter.create<arith::ShRUIOp>(
        loc, i32Src, rewriter.create<arith::ConstantOp>(loc, shiftValAttr));
    Value i16Res = rewriter.create<arith::TruncIOp>(loc, toInt16(src.getType()),
                                                    shiftedSrc);
    Value res = rewriter.create<arith::BitcastOp>(loc, op.getType(), i16Res);
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct DecomposeFpConversions
    : public triton::impl::DecomposeFpConversionsBase<DecomposeFpConversions> {
  using DecomposeFpConversionsBase::DecomposeFpConversionsBase;

  DecomposeFpConversions() : DecomposeFpConversionsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<Fp32ToBf16Conversion>(context);

    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createDecomposeFpConversions() {
  return std::make_unique<DecomposeFpConversions>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
