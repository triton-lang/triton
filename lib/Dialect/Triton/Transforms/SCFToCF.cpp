#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define GEN_PASS_DEF_TRITONSCFTOCF
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

using namespace mlir;
using namespace mlir::scf;

// While loop lowering patterns forked from MLIR lowering. ForOp already has the
// propagation.
// TODO: Upstream llvm loop attribute propagation and remove this pass.
namespace {
struct SCFToCFPass : public ::impl::TritonSCFToCFBase<SCFToCFPass> {
  void runOnOperation() override;
};

/// Create a CFG subgraph for this loop construct. The regions of the loop need
/// not be a single block anymore (for example, if other SCF constructs that
/// they contain have been already converted to CFG), but need to be single-exit
/// from the last block of each region. The operations following the original
/// WhileOp are split into a new continuation block. Both regions of the WhileOp
/// are inlined, and their terminators are rewritten to organize the control
/// flow implementing the loop as follows.
///
///      +---------------------------------+
///      |   <code before the WhileOp>     |
///      |   cf.br ^before(%operands...)      |
///      +---------------------------------+
///             |
///  -------|   |
///  |      v   v
///  |   +--------------------------------+
///  |   | ^before(%bargs...):            |
///  |   |   %vals... = <some payload>    |
///  |   +--------------------------------+
///  |                   |
///  |                  ...
///  |                   |
///  |   +--------------------------------+
///  |   | ^before-last:
///  |   |   %cond = <compute condition>  |
///  |   |   cf.cond_br %cond,               |
///  |   |        ^after(%vals...), ^cont |
///  |   +--------------------------------+
///  |          |               |
///  |          |               -------------|
///  |          v                            |
///  |   +--------------------------------+  |
///  |   | ^after(%aargs...):             |  |
///  |   |   <body contents>              |  |
///  |   +--------------------------------+  |
///  |                   |                   |
///  |                  ...                  |
///  |                   |                   |
///  |   +--------------------------------+  |
///  |   | ^after-last:                   |  |
///  |   |   %yields... = <some payload>  |  |
///  |   |   cf.br ^before(%yields...)       |  |
///  |   +--------------------------------+  |
///  |          |                            |
///  |-----------        |--------------------
///                      v
///      +--------------------------------+
///      | ^cont:                         |
///      |   <code after the WhileOp>     |
///      |   <%vals from 'before' region  |
///      |          visible by dominance> |
///      +--------------------------------+
///
/// Values are communicated between ex-regions (the groups of blocks that used
/// to form a region before inlining) through block arguments of their
/// entry blocks, which are visible in all other dominated blocks. Similarly,
/// the results of the WhileOp are defined in the 'before' region, which is
/// required to have a single existing block, and are therefore accessible in
/// the continuation block due to dominance.
struct WhileLowering : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp whileOp,
                                PatternRewriter &rewriter) const override;
};

/// Optimized version of the above for the case of the "after" region merely
/// forwarding its arguments back to the "before" region (i.e., a "do-while"
/// loop). This avoid inlining the "after" region completely and branches back
/// to the "before" entry instead.
struct DoWhileLowering : public OpRewritePattern<WhileOp> {
  using OpRewritePattern<WhileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WhileOp whileOp,
                                PatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult WhileLowering::matchAndRewrite(WhileOp whileOp,
                                             PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = whileOp.getLoc();

  // Split the current block before the WhileOp to create the inlining point.
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *continuation =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  // Inline both regions.
  Block *after = whileOp.getAfterBody();
  Block *before = whileOp.getBeforeBody();
  rewriter.inlineRegionBefore(whileOp.getAfter(), continuation);
  rewriter.inlineRegionBefore(whileOp.getBefore(), after);

  // Branch to the "before" region.
  rewriter.setInsertionPointToEnd(currentBlock);
  rewriter.create<cf::BranchOp>(loc, before, whileOp.getInits());

  // Replace terminators with branches. Assuming bodies are SESE, which holds
  // given only the patterns from this file, we only need to look at the last
  // block. This should be reconsidered if we allow break/continue in SCF.
  rewriter.setInsertionPointToEnd(before);
  auto condOp = cast<ConditionOp>(before->getTerminator());
  SmallVector<Value> args = llvm::to_vector(condOp.getArgs());
  rewriter.replaceOpWithNewOp<cf::CondBranchOp>(condOp, condOp.getCondition(),
                                                after, condOp.getArgs(),
                                                continuation, ValueRange());

  rewriter.setInsertionPointToEnd(after);
  auto yieldOp = cast<scf::YieldOp>(after->getTerminator());
  auto latch = rewriter.replaceOpWithNewOp<cf::BranchOp>(yieldOp, before,
                                                         yieldOp.getResults());

  // Let the CondBranchOp carry the LLVM attributes from the ForOp, such as the
  // llvm.loop_annotation attribute.
  SmallVector<NamedAttribute> llvmAttrs;
  llvm::copy_if(whileOp->getAttrs(), std::back_inserter(llvmAttrs),
                [](auto attr) {
                  return isa<LLVM::LLVMDialect>(attr.getValue().getDialect());
                });
  latch->setDiscardableAttrs(llvmAttrs);
  // Replace the op with values "yielded" from the "before" region, which are
  // visible by dominance.
  rewriter.replaceOp(whileOp, args);

  return success();
}

LogicalResult
DoWhileLowering::matchAndRewrite(WhileOp whileOp,
                                 PatternRewriter &rewriter) const {
  Block &afterBlock = *whileOp.getAfterBody();
  if (!llvm::hasSingleElement(afterBlock))
    return rewriter.notifyMatchFailure(whileOp,
                                       "do-while simplification applicable "
                                       "only if 'after' region has no payload");

  auto yield = dyn_cast<scf::YieldOp>(&afterBlock.front());
  if (!yield || yield.getResults() != afterBlock.getArguments())
    return rewriter.notifyMatchFailure(whileOp,
                                       "do-while simplification applicable "
                                       "only to forwarding 'after' regions");

  // Split the current block before the WhileOp to create the inlining point.
  OpBuilder::InsertionGuard guard(rewriter);
  Block *currentBlock = rewriter.getInsertionBlock();
  Block *continuation =
      rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());

  // Only the "before" region should be inlined.
  Block *before = whileOp.getBeforeBody();
  rewriter.inlineRegionBefore(whileOp.getBefore(), continuation);

  // Branch to the "before" region.
  rewriter.setInsertionPointToEnd(currentBlock);
  auto latch = rewriter.create<cf::BranchOp>(whileOp.getLoc(), before,
                                             whileOp.getInits());

  // Loop around the "before" region based on condition.
  rewriter.setInsertionPointToEnd(before);
  auto condOp = cast<ConditionOp>(before->getTerminator());
  SmallVector<Value> args = llvm::to_vector(condOp.getArgs());
  rewriter.replaceOpWithNewOp<cf::CondBranchOp>(condOp, condOp.getCondition(),
                                                before, condOp.getArgs(),
                                                continuation, ValueRange());

  // Let the CondBranchOp carry the LLVM attributes from the ForOp, such as the
  // llvm.loop_annotation attribute.
  SmallVector<NamedAttribute> llvmAttrs;
  llvm::copy_if(whileOp->getAttrs(), std::back_inserter(llvmAttrs),
                [](auto attr) {
                  return isa<LLVM::LLVMDialect>(attr.getValue().getDialect());
                });
  latch->setDiscardableAttrs(llvmAttrs);

  // Replace the op with values "yielded" from the "before" region, which are
  // visible by dominance.
  rewriter.replaceOp(whileOp, args);

  return success();
}

void SCFToCFPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  // Give our patterns higher benefits so that they get picked up instead of the
  // MLIR one.
  patterns.add<WhileLowering>(&getContext(), /*benefit=*/3);
  patterns.add<DoWhileLowering>(&getContext(), /*benefit=*/4);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);

  // Configure conversion to lower out SCF operations.
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ForallOp, scf::ForOp, scf::IfOp, scf::IndexSwitchOp,
                      scf::ParallelOp, scf::WhileOp, scf::ExecuteRegionOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

namespace mlir::triton {
std::unique_ptr<mlir::Pass> createTritonSCFToCF() {
  return std::make_unique<SCFToCFPass>();
}
} // namespace mlir::triton
