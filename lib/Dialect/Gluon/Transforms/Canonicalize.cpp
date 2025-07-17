#include "triton/Dialect/Gluon/Transforms/Passes.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace triton;
namespace ttg = triton::gpu;
namespace ttng = triton::nvidia_gpu;
namespace gluon = mlir::triton::gluon;

namespace mlir::triton::gluon {
#define GEN_PASS_DEF_GLUONCANONICALIZE
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"
} // namespace mlir::triton::gluon

namespace {
struct Canonicalize : public gluon::impl::GluonCanonicalizeBase<Canonicalize> {
  void runOnOperation() override;
};
} // namespace

void Canonicalize::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(&getContext());

  // Populate `arith` and `scf` canonicalizers.
  ctx->getLoadedDialect<arith::ArithDialect>()->getCanonicalizationPatterns(
      patterns);
  ctx->getLoadedDialect<scf::SCFDialect>()->getCanonicalizationPatterns(
      patterns);
  ctx->getLoadedDialect<cf::ControlFlowDialect>()->getCanonicalizationPatterns(
      patterns);
  populateForOpDeadArgumentElimination(patterns);

  // Populate select Triton canonicalization patterns. The important patterns to
  // EXCLUDE are those that modify layouts, especially `ConvertLayoutOp`
  // patterns.
  LoadOp::getCanonicalizationPatterns(patterns, ctx);
  StoreOp::getCanonicalizationPatterns(patterns, ctx);
  BroadcastOp::getCanonicalizationPatterns(patterns, ctx);
  ExpandDimsOp::getCanonicalizationPatterns(patterns, ctx);
  ttg::WarpSpecializeOp::getCanonicalizationPatterns(patterns, ctx);

  (void)applyPatternsGreedily(getOperation(), std::move(patterns));
}
