#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"

#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace triton;

namespace mlir::triton::gluon {
#define GEN_PASS_DEF_GLUONSIMPLIFYCONTROLFLOW
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"
} // namespace mlir::triton::gluon

namespace {
struct SimplifyControlFlow
    : public gluon::impl::GluonSimplifyControlFlowBase<SimplifyControlFlow> {
  void runOnOperation() override;
};
} // namespace

void SimplifyControlFlow::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(&getContext());

  // Populate `scf` and `cf` canonicalizers.
  ctx->getLoadedDialect<scf::SCFDialect>()->getCanonicalizationPatterns(
      patterns);
  ctx->getLoadedDialect<cf::ControlFlowDialect>()->getCanonicalizationPatterns(
      patterns);
  for (mlir::RegisteredOperationName op : ctx->getRegisteredOperationsByDialect(
           scf::SCFDialect::getDialectNamespace()))
    op.getCanonicalizationPatterns(patterns, ctx);
  for (mlir::RegisteredOperationName op : ctx->getRegisteredOperationsByDialect(
           cf::ControlFlowDialect::getDialectNamespace()))
    op.getCanonicalizationPatterns(patterns, ctx);
  populateForOpDeadArgumentElimination(patterns);

  GreedyRewriteConfig config;
  // This is intended to run before AutoLayouts are resolved, in which case
  // CSEing constants can lead to additional layout conflicts.
  config.enableConstantCSE(false);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
}
