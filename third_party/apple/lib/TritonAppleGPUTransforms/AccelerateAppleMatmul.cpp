// AccelerateAppleMatmul: rewrite tt.dot ops to use AppleMmaEncodingAttr
//
// This is the Triton GPU lowering pass that replaces the generic
// BlockedEncoding on dot ops with AppleMmaEncoding, enabling
// simdgroup_multiply_accumulate code generation.
//
// Mirrors AccelerateAMDMatmul.cpp (BlockedToMFMA) for Apple.
//
// Pipeline position:
//   make_ttgir: TritonGPU IR → Apple MMA tiled TritonGPU IR
//
// What it does:
//   1. Find all tt.dot ops
//   2. Check that element types are supported (f16, bf16, f32)
//   3. Replace output encoding: BlockedEncoding → AppleMmaEncoding
//   4. Insert ConvertLayoutOp on operands to match expected input layout
//   5. Insert ConvertLayoutOp on output to convert back to user's expected layout

#include "Dialect/TritonAppleGPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

#define GEN_PASS_DEF_ACCELERATEAPPLEMATMUL
#include "TritonAppleGPUTransforms/Passes.h.inc"

namespace tt  = mlir::triton;
namespace ttg = mlir::triton::gpu;
using namespace mlir;
using namespace mlir::triton::applegpu;

namespace {

// Determine warpsPerCTA for a given dot op shape and total warp count.
// Apple simdgroup tile = 8x8, so:
//   warpsM = ceil(M / 8), warpsN = ceil(N / 8), capped by numWarps.
SmallVector<unsigned> warpsPerTileApple(int64_t M, int64_t N, int numWarps) {
    unsigned warpsM = std::max(1LL, M / 8);
    unsigned warpsN = std::max(1LL, N / 8);

    // Clamp to numWarps budget (prefer square allocation)
    while (warpsM * warpsN > (unsigned)numWarps) {
        if (warpsM > warpsN) warpsM /= 2;
        else warpsN /= 2;
    }
    return {warpsM, warpsN};
}

// Check if element type is supported by simdgroup_multiply_accumulate
bool isSupportedDotType(mlir::Type elemTy) {
    return elemTy.isF16() || elemTy.isBF16() || elemTy.isF32();
}

// Pattern: BlockedEncoding tt.dot → AppleMmaEncoding tt.dot
struct BlockedToAppleMma : public OpRewritePattern<tt::DotOp> {
    int numWarps;

    BlockedToAppleMma(MLIRContext *ctx, int numWarps, PatternBenefit benefit = 1)
        : OpRewritePattern(ctx, benefit), numWarps(numWarps) {}

    LogicalResult matchAndRewrite(tt::DotOp dot,
                                  PatternRewriter &rewriter) const override {
        auto ctx = dot.getContext();
        auto cType = cast<RankedTensorType>(dot.getC().getType());
        auto aType = cast<RankedTensorType>(dot.getA().getType());

        // Skip — keep BlockedEncoding on dot ops.
        // AppleMmaEncoding rewrite causes verifier failures (Triton's
        // DotOp verifier only accepts known parent layouts for DotOperandEncoding,
        // and DialectInferLayoutInterface is not yet registered for Apple dialect).
        // The DotOpToLLVM conversion handles BlockedEncoding dot ops directly.
        (void)cType;
        (void)aType;
        return failure();

        // Check supported element types
        if (!isSupportedDotType(aType.getElementType()))
            return failure();

        auto shape = cType.getShape();
        if (shape.size() != 2) return failure();

        int64_t M = shape[0], N = shape[1];

        // Create AppleMmaEncoding
        auto wpc = warpsPerTileApple(M, N, numWarps);
        auto mmaEnc = AppleMmaEncodingAttr::get(ctx, wpc);

        auto newCType = RankedTensorType::get(shape, cType.getElementType(), mmaEnc);

        // Keep A, B with AppleMmaEncoding (same as C) — DotOperandEncoding
        // not used because Triton's verifier doesn't know AppleMmaEncoding
        // as a valid DotOperandEncoding parent. The DotOpToLLVM conversion
        // handles unblocking via the TG scatter/gather path.
        auto newAType = RankedTensorType::get(aType.getShape(),
                                              aType.getElementType(), mmaEnc);
        auto newBType = RankedTensorType::get(
            cast<RankedTensorType>(dot.getB().getType()).getShape(),
            cast<RankedTensorType>(dot.getB().getType()).getElementType(), mmaEnc);

        // Insert layout conversions for operands
        auto loc = dot.getLoc();
        Value newA = ttg::ConvertLayoutOp::create(rewriter, loc, newAType, dot.getA());
        Value newB = ttg::ConvertLayoutOp::create(rewriter, loc, newBType, dot.getB());
        Value newC = ttg::ConvertLayoutOp::create(rewriter, loc, newCType, dot.getC());

        // Create new dot op with AppleMma encoding
        auto newDot = tt::DotOp::create(rewriter,
            loc, newCType, newA, newB, newC,
            dot.getInputPrecisionAttr(), dot.getMaxNumImpreciseAccAttr());

        // Convert output back to original encoding
        auto result = ttg::ConvertLayoutOp::create(rewriter,
            loc, cType, newDot.getResult());

        rewriter.replaceOp(dot, result.getResult());
        return success();
    }
};

// The pass
struct AccelerateAppleMatmul
    : public ::impl::AccelerateAppleMatmulBase<AccelerateAppleMatmul> {

    void runOnOperation() override {
        auto mod = getOperation();

        // Get numWarps from module attribute
        int numWarps = ttg::lookupNumWarps(mod);

        RewritePatternSet patterns(&getContext());
        patterns.add<BlockedToAppleMma>(&getContext(), numWarps);

        if (failed(applyPatternsGreedily(mod, std::move(patterns))))
            signalPassFailure();
    }
};

} // anonymous namespace

namespace mlir::triton::applegpu {
std::unique_ptr<mlir::Pass> createAccelerateAppleMatmulPass() {
    return ::createAccelerateAppleMatmul();
}
} // namespace mlir::triton::applegpu
