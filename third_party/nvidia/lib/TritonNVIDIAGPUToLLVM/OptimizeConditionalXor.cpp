#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton {

#define GEN_PASS_DEF_NVIDIAOPTIMIZECONDITIONALXOR
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"

namespace {

// The shared one-hot reduction pass produces ordinary gathers. After LLVM
// conversion and canonicalization expose their scalar results, fuse conditional
// XORs that consume them without teaching individual lowerings about their
// users.
class FuseConditionalXorPattern : public OpRewritePattern<LLVM::XOrOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::XOrOp accumulation,
                                PatternRewriter &rewriter) const override {
    if (!accumulation.getType().isInteger(32))
      return failure();

    auto contribution = matchContribution(accumulation.getRhs());
    Value accumulator = accumulation.getLhs();
    if (!contribution) {
      contribution = matchContribution(accumulation.getLhs());
      accumulator = accumulation.getRhs();
    }
    if (!contribution)
      return failure();

    // Select masks a poison basis on its false arm; inline assembly eagerly
    // consumes every operand. Freeze only this use so that a false condition
    // still contributes zero without changing the basis's other masked uses.
    auto loc = accumulation.getLoc();
    Value frozenBasis =
        LLVM::FreezeOp::create(rewriter, loc, contribution.getTrueValue());
    Value signMask = LLVM::SExtOp::create(rewriter, loc, accumulation.getType(),
                                          contribution.getCondition());
    auto fused = LLVM::InlineAsmOp::create(
        rewriter, loc, accumulation.getType(),
        ValueRange{accumulator, frozenBasis, signMask},
        "lop3.b32 $0, $1, $2, $3, 0x78;", "=r,r,r,r",
        /*has_side_effects=*/false, /*is_align_stack=*/false,
        LLVM::TailCallKind::None,
        LLVM::AsmDialectAttr::get(getContext(), LLVM::AsmDialect::AD_ATT),
        ArrayAttr{});
    rewriter.replaceOp(accumulation, fused.getRes());
    rewriter.eraseOp(contribution);
    return success();
  }

private:
  static LLVM::SelectOp matchContribution(Value value) {
    auto select = value.getDefiningOp<LLVM::SelectOp>();
    if (!select || !select->hasOneUse() ||
        !select.getCondition().getType().isInteger(1) ||
        !matchPattern(select.getFalseValue(), m_Zero()))
      return {};

    auto *basis = select.getTrueValue().getDefiningOp();
    if (!basis || !basis->getAttrOfType<UnitAttr>(gpu::AttrOneHotXorReduction))
      return {};
    return select;
  }
};

class NvidiaOptimizeConditionalXorPass
    : public impl::NvidiaOptimizeConditionalXorBase<
          NvidiaOptimizeConditionalXorPass> {
public:
  void runOnOperation() override {
    auto module = getOperation();
    auto target = module->getAttrOfType<StringAttr>(gpu::AttrTargetName);
    if (!target || !target.getValue().starts_with("cuda:"))
      return;

    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseConditionalXorPattern>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns), config)))
      return signalPassFailure();

    // This provenance is only needed by this pass, including when the gather
    // has no conditional XOR user or the contribution cannot be fused.
    module.walk(
        [](Operation *op) { op->removeAttr(gpu::AttrOneHotXorReduction); });
  }
};

} // namespace
} // namespace mlir::triton
