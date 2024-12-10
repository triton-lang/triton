#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CANONICALIZE
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Fold transfer read and the following shape cast that removes heading
// dimensions with size 1.
struct FoldReadShapeCast : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (!op->hasOneUse())
      return failure();

    auto permMap = op.getPermutationMap();
    if (!permMap.isMinorIdentity())
      return failure();

    auto reshape = dyn_cast<vector::ShapeCastOp>(*op->user_begin());
    if (!reshape)
      return failure();

    VectorType ty = cast<VectorType>(op.getType());
    VectorType dstTy = cast<VectorType>(reshape.getType());
    if (ty.getRank() <= dstTy.getRank())
      return failure();

    // Check all removed dimensions have size 1.
    if (!all_of(drop_end(ty.getShape(), dstTy.getRank()),
                [](int64_t val) { return val == 1; }))
      return failure();

    // Check shape prefix matches the resulting type.
    if (!equal(drop_begin(ty.getShape(), ty.getRank() - dstTy.getRank()),
               dstTy.getShape()))
      return failure();

    auto inBounds = op.getInBounds();
    if (std::any_of(inBounds.begin(), inBounds.end() - dstTy.getRank(),
                    [](Attribute attr) {
                      return !cast<mlir::BoolAttr>(attr).getValue();
                    }))
      return failure();

    // Fold read and shape cast into a single read.
    auto newPermMap = permMap.getMinorIdentityMap(
        permMap.getNumDims(), dstTy.getRank(), getContext());
    auto newInBounds = rewriter.getArrayAttr(SmallVector<Attribute>(drop_begin(
        op.getInBounds().getValue(), ty.getRank() - dstTy.getRank())));
    auto newRead = rewriter.create<vector::TransferReadOp>(
        loc, dstTy, op.getSource(), op.getIndices(), newPermMap,
        op.getPadding(), op.getMask(), newInBounds);
    rewriter.replaceOp(reshape, newRead);
    rewriter.eraseOp(op);

    return success();
  }
};

struct Canonicalize : public triton::cpu::impl::CanonicalizeBase<Canonicalize> {
  Canonicalize() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldReadShapeCast>(context);

    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalize() {
  return std::make_unique<Canonicalize>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
